"""
Loop closure for DPVO.
Merged from loop_closure/{optim_utils,long_term,retrieval/*}.py into a single flat module.
"""

import logging
import os
import time
from multiprocessing import Process, Queue, Value, Pool
from shutil import copytree
from tempfile import TemporaryDirectory

import cv2
import cuda_ba
import kornia as K
import kornia.feature as KF
import numba as nb
import numpy as np
import pypose as pp
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from einops import asnumpy, parse_shape, rearrange, repeat
from scipy.spatial.transform import Rotation as R
from torch_scatter import scatter_max

from . import fastba
from . import projective_ops as pops
from .lietorch import SE3

log = logging.getLogger("motioncapture")


# ---------------------------------------------------------------------------
# optim_utils
# ---------------------------------------------------------------------------

def make_pypose_Sim3(rot, t, s):
    q = R.from_matrix(rot).as_quat()
    data = np.concatenate([t, q, np.array(s).reshape((1,))])
    return pp.Sim3(data)

def SE3_to_Sim3(x: pp.SE3):
    out = torch.cat((x.data, torch.ones_like(x.data[...,:1])), dim=-1)
    return pp.Sim3(out)

@nb.njit(cache=True)
def _format(es):
    return np.asarray(es, dtype=np.int64).reshape((-1, 2))[1:]

@nb.njit(cache=True)
def reduce_edges(flow_mag, ii, jj, max_num_edges, nms):
    es = [(-1, -1)]

    if ii.size == 0:
        return _format(es)

    Ni, Nj = (ii.max()+1), (jj.max()+1)
    ignore_lookup = np.zeros((Ni, Nj), dtype=nb.bool_)

    idxs = np.argsort(flow_mag)
    for idx in idxs: # edge index

        if len(es) > max_num_edges:
            break

        i = ii[idx]
        j = jj[idx]
        mag = flow_mag[idx]

        if ((j - i) < 30):
            continue

        if mag >= 1000: # i.e., inf
            continue

        if ignore_lookup[i, j]:
            continue

        es.append((i, j))

        for di in range(-nms, nms+1):
            i1 = i + di

            if 0 <= i1 < Ni:
                ignore_lookup[i1, j] = True

    return _format(es)


def batch_jacobian(func, x):
  def _func_sum(*x):
    return func(*x).sum(dim=0)
  _, b, c = torch.autograd.functional.jacobian(_func_sum, x, vectorize=True)
  return rearrange(torch.stack((b,c)), 'N O B I -> N B O I', N=2)

def _residual(C, Gi, Gj):
    assert parse_shape(C, 'N _') == parse_shape(Gi, 'N _') == parse_shape(Gj, 'N _')
    out = C @ pp.Exp(Gi) @ pp.Exp(Gj).Inv()
    return out.Log().tensor()

def residual(Ginv, input_poses, dSloop, ii, jj, jacobian=False):

    # prep
    device = Ginv.device
    assert parse_shape(input_poses, '_ d') == dict(d=7)
    pred_inv_poses = SE3_to_Sim3(input_poses).Inv()

    # free variables
    n, _ = pred_inv_poses.shape
    kk = torch.arange(1, n, device=device)
    ll = kk-1

    # constants
    Ti = pred_inv_poses[kk]
    Tj = pred_inv_poses[ll]
    dSij = Tj @ Ti.Inv()

    constants = torch.cat((dSij, dSloop), dim=0)
    iii = torch.cat((kk, ii))
    jjj = torch.cat((ll, jj))
    resid = _residual(constants, Ginv[iii], Ginv[jjj])

    if not jacobian:
        return resid

    J_Ginv_i, J_Ginv_j = batch_jacobian(_residual, (constants, Ginv[iii], Ginv[jjj]))
    return resid, (J_Ginv_i, J_Ginv_j, iii, jjj)

def run_DPVO_PGO(pred_poses, loop_poses, loop_ii, loop_jj, queue):
    final_est = perform_updates(pred_poses, loop_poses, loop_ii, loop_jj, iters=30)

    safe_i = loop_ii.max().item() + 1
    aa = SE3_to_Sim3(pred_poses.cpu())
    final_est = (aa[[safe_i]] * final_est[[safe_i]].Inv()) * final_est
    output = final_est[:safe_i]
    queue.put(output)

def perform_updates(input_poses, dSloop, ii_loop, jj_loop, iters, ep=0.0, lmbda=1e-6, fix_opt_window=False):
    """ Run the Levenberg Marquardt algorithm """

    input_poses = input_poses.clone()

    if fix_opt_window:
        freen = torch.cat((ii_loop, jj_loop)).max().item() + 1
    else:
        freen = -1

    Ginv = SE3_to_Sim3(input_poses).Inv().Log()

    residual_history = []

    for itr in range(iters):
        resid, (J_Ginv_i, J_Ginv_j, iii, jjj) = residual(Ginv, input_poses, dSloop, ii_loop, jj_loop, jacobian=True)
        residual_history.append(resid.square().mean().item())
        delta_pose, = cuda_ba.solve_system(J_Ginv_i, J_Ginv_j, iii, jjj, resid, ep, lmbda, freen)
        assert Ginv.shape == delta_pose.shape
        Ginv_tmp = Ginv + delta_pose

        new_resid = residual(Ginv_tmp, input_poses, dSloop, ii_loop, jj_loop)
        if new_resid.square().mean() < residual_history[-1]:
            Ginv = Ginv_tmp
            lmbda /= 2
        else:
            lmbda *= 2

        if (residual_history[-1] < 1e-5) and (itr >= 4) and ((residual_history[-5] / residual_history[-1]) < 1.5):
            break

    return pp.Exp(Ginv).Inv()


# ---------------------------------------------------------------------------
# retrieval_dbow
# ---------------------------------------------------------------------------

NMS = 50
RAD = 50

def _dbow_loop(in_queue, out_queue, vocab_path, ready):
    """ Run DBoW retrieval """
    import dpretrieval
    dbow = dpretrieval.DPRetrieval(vocab_path, 50)
    ready.value = 1
    while True:
        n, image = in_queue.get()
        dbow.insert_image(image)
        q = dbow.query(n)
        out_queue.put((n, q))

class RetrievalDBOW:

    def __init__(self, vocab_path="ORBvoc.txt"):
        if not os.path.exists(vocab_path):
            raise FileNotFoundError("""Missing the ORB vocabulary. Please download and un-tar it from """
                                  """https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/master/Vocabulary/ORBvoc.txt.tar.gz"""
                                  f""" and place it in DPVO/""")

        # Store a record of saved and unsaved images
        self.image_buffer = {}
        self.stored_indices = np.zeros(100000, dtype=bool)

        # Keep track of detected and closed loops
        self.prev_loop_closes = []
        self.found = []

        # Run DBoW in a separate process
        self.in_queue = Queue(maxsize=20)
        self.out_queue = Queue(maxsize=20)
        ready = Value('i', 0)
        self.proc = Process(target=_dbow_loop, args=(self.in_queue, self.out_queue, vocab_path, ready))
        self.proc.start()
        self.being_processed = 0
        while not ready.value:
            time.sleep(0.01)

    def keyframe(self, k):
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n-1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c):
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                img = self.image_buffer.pop(n)
                self.in_queue.put((n, img))
                self.stored_indices[n] = True
                self.being_processed += 1

    def confirm_loop(self, i, j):
        assert i > j
        self.prev_loop_closes.append((i, j))

    def _repetition_check(self, idx, num_repeat):
        if (len(self.found) < num_repeat):
            return
        latest = self.found[-num_repeat:]
        (b, _), (i, j), _ = latest
        if (1 + idx - b) == num_repeat:
            return (i, max(j,1))

    def detect_loop(self, thresh, num_repeat=1):
        while self.being_processed > 0:
            x = self._detect_loop(thresh, num_repeat)
            if x is not None:
                return x

    def _detect_loop(self, thresh, num_repeat=1):
        assert self.being_processed > 0
        i, (score, j, _) = self.out_queue.get()
        self.being_processed -= 1
        if score < thresh:
            return
        assert i > j

        dists_sq = [(np.square(i - a) + np.square(j - b)) for a,b in self.prev_loop_closes]
        if min(dists_sq, default=np.inf) < np.square(NMS):
            return

        self.found.append((i, j))
        return self._repetition_check(i, num_repeat)

    def __call__(self, image, n):
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, '_ _ RGB') == dict(RGB=3)
        self.image_buffer[n] = image

    def close(self):
        self.proc.terminate()
        self.proc.join()


# ---------------------------------------------------------------------------
# image_cache
# ---------------------------------------------------------------------------

IMEXT = '.jpeg'
JPEG_QUALITY = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
BLANK = np.zeros((500,500,3), dtype=np.uint8)

class ImageCache:

    def __init__(self):
        self.image_buffer = {}
        self.tmpdir = TemporaryDirectory()
        self.stored_indices = np.zeros(100000, dtype=bool)
        self.writer_pool = Pool(processes=1)
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [f"{self.tmpdir.name}/warmup.png", BLANK, JPEG_QUALITY])
        self._wait()

    def __call__(self, image, n):
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert parse_shape(image, '_ _ RGB') == dict(RGB=3)
        self.image_buffer[n] = image

    def _wait(self):
        self.write_result.wait()

    def _write_image(self, i):
        img = self.image_buffer.pop(i)
        filepath = f"{self.tmpdir.name}/{i:08d}{IMEXT}"
        assert not os.path.exists(filepath)
        self._wait()
        self.write_result = self.writer_pool.apply_async(cv2.imwrite, [filepath, img, JPEG_QUALITY])

    def load_frames(self, idxs, device):
        self._wait()
        assert np.all(self.stored_indices[idxs])
        frame_list = [f"{self.tmpdir.name}/{i:08d}{IMEXT}" for i in idxs]
        assert all(map(os.path.exists, frame_list))
        image_list = [cv2.imread(f) for f in frame_list]
        return K.utils.image_list_to_tensor(image_list).to(device=device)

    def keyframe(self, k):
        tmp = dict(self.image_buffer)
        self.image_buffer.clear()
        for n, v in tmp.items():
            if n != k:
                key = (n-1) if (n > k) else n
                self.image_buffer[key] = v

    def save_up_to(self, c):
        for n in list(self.image_buffer):
            if n <= c:
                assert not self.stored_indices[n]
                self._write_image(n)
                self.stored_indices[n] = True

    def close(self):
        self._wait()
        self.tmpdir.cleanup()
        self.writer_pool.close()


# ---------------------------------------------------------------------------
# long_term loop closure
# ---------------------------------------------------------------------------

class LongTermLoopClosure:

    def __init__(self, cfg, patchgraph):
        self.cfg = cfg
        self.device = patchgraph.device

        # Data structures to manage retrieval
        self.retrieval = RetrievalDBOW()
        self.imcache = ImageCache()

        # Process to run PGO in parallel
        self.lc_pool = mp.Pool(processes=1)
        self.lc_process = self.lc_pool.apply_async(os.getpid)
        self.manager = mp.Manager()
        self.result_queue = self.manager.Queue()
        self.lc_in_progress = False

        # Patch graph + loop edges
        self.pg = patchgraph
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)

        self.lc_count = 0

        # warmup the jit compiler
        ransac_umeyama(np.random.randn(3,3), np.random.randn(3,3), iterations=200, threshold=0.01)

        self.detector = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.matcher = KF.LightGlue("disk").to(self.device).eval()

    def detect_keypoints(self, images, num_features=2048):
        _, _, h, w = images.shape
        wh = torch.tensor([w, h]).view(1, 2).float().to(self.device)
        features = self.detector(images, num_features, pad_if_not_divisible=True, window_size=15, score_threshold=40.0)
        return [{
            "keypoints": f.keypoints[None],
            "descriptors": f.descriptors[None],
            "image_size": wh
        } for f in features]


    def __call__(self, img, n):
        img_np = K.tensor_to_image(img)
        self.retrieval(img_np, n)
        self.imcache(img_np, n)

    def keyframe(self, k):
        self.retrieval.keyframe(k)
        self.imcache.keyframe(k)

    def estimate_3d_keypoints(self, i):
        """ Detect, match and triangulate 3D points """

        image_orig = self.imcache.load_frames([i-1,i,i+1], self.pg.intrinsics.device)
        image = image_orig.float() / 255
        fl = self.detect_keypoints(image)

        trajectories = torch.full((2048, 3), -1, device=self.device, dtype=torch.long)
        trajectories[:,1] = torch.arange(2048)

        out = self.matcher({"image0": fl[0], "image1": fl[1]})
        i0, i1 = out["matches"][0].mT
        trajectories[i1, 0] = i0

        out = self.matcher({"image0": fl[2], "image1": fl[1]})
        i2, i1 = out["matches"][0].mT
        trajectories[i1, 2] = i2

        trajectories = trajectories[torch.randperm(2048)]
        trajectories = trajectories[trajectories.min(dim=1).values >= 0]

        a,b,c = trajectories.mT
        n, _ = trajectories.shape
        kps0 = fl[0]['keypoints'][:,a]
        kps1 = fl[1]['keypoints'][:,b]
        kps2 = fl[2]['keypoints'][:,c]

        desc1 = fl[1]['descriptors'][:,b]
        image_size = fl[1]["image_size"]

        kk = torch.arange(n).to(self.device).repeat(2)
        ii = torch.ones(2*n, device=self.device, dtype=torch.long)
        jj = torch.zeros(2*n, device=self.device, dtype=torch.long)
        jj[n:] = 2

        true_disp = self.pg.patches_[i,:,2,1,1].median()
        patches = torch.cat((kps1, torch.ones(1, n, 1, device=self.device) * true_disp), dim=-1)
        patches = repeat(patches, '1 n uvd -> 1 n uvd 3 3', uvd=3)
        target = rearrange(torch.stack((kps0, kps2)), 'ot 1 n uv -> 1 (ot n) uv', uv=2, n=n, ot=2)
        weight = torch.ones_like(target)

        poses = self.pg.poses[:,i-1:i+2].clone()
        intrinsics = self.pg.intrinsics[:,i-1:i+2].clone() * 4

        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:,:,1,1]
        residual_val = (coords - target).norm(dim=-1).squeeze(0)

        lmbda = torch.as_tensor([1e-3], device=self.device)
        fastba.BA(poses, patches, intrinsics,
            target, weight, lmbda, ii, jj, kk, 3, 3, M=-1, iterations=6, eff_impl=False)

        coords = pops.transform(SE3(poses), patches, intrinsics, ii, jj, kk)
        coords = coords[:,:,1,1]
        residual_val = (coords - target).norm(dim=-1).squeeze(0)
        assert residual_val.numel() == 2*n
        mask = scatter_max(residual_val, kk)[0] < 2

        points = pops.iproj(patches, intrinsics[:,torch.ones(n, device=self.device, dtype=torch.long)])
        points = (points[...,1,1,:3] / points[...,1,1,3:])

        return points[:,mask].squeeze(0), {"keypoints": kps1[:,mask], "descriptors": desc1[:,mask], "image_size": image_size}

    def attempt_loop_closure(self, n):
        if self.lc_in_progress:
            return

        cands = self.retrieval.detect_loop(thresh=self.cfg.LOOP_RETR_THRESH, num_repeat=self.cfg.LOOP_CLOSE_WINDOW_SIZE)
        if cands is not None:
            i, j = cands

            lc_result = self.close_loop(i, j, n)
            self.lc_count += int(lc_result)

            if lc_result:
                self.retrieval.confirm_loop(i, j)
            self.retrieval.found.clear()

        self.retrieval.save_up_to(n - self.cfg.REMOVAL_WINDOW - 2)
        self.imcache.save_up_to(n - self.cfg.REMOVAL_WINDOW - 1)

    def terminate(self, n):
        self.retrieval.save_up_to(n-1)
        self.imcache.save_up_to(n-1)
        self.attempt_loop_closure(n)
        if self.lc_in_progress:
            self.lc_callback(skip_if_empty=False)
        self.lc_process.get()
        self.imcache.close()
        self.lc_pool.close()
        self.retrieval.close()
        log.info("LC COUNT: %d", self.lc_count)


    def _rescale_deltas(self, s):
        tstamp_2_rescale = {}
        for i in range(self.pg.n):
            tstamp_2_rescale[self.pg.tstamps_[i]] = s[i]

        for t, (t0, dP) in self.pg.delta.items():
            t_src = t
            while t_src in self.pg.delta:
                t_src, _ = self.pg.delta[t_src]
            s1 = tstamp_2_rescale[t_src]
            self.pg.delta[t] = (t0, dP.scale(s1))

    def lc_callback(self, skip_if_empty=True):
        if skip_if_empty and self.result_queue.empty():
            return
        self.lc_in_progress = False
        final_est = self.result_queue.get()
        safe_i, _ = final_est.shape
        res, s = final_est.tensor().to(self.device).split([7,1], dim=1)
        s1 = torch.ones(self.pg.n, device=s.device)
        s1[:safe_i] = s.squeeze()

        self.pg.poses_[:safe_i] = SE3(res).inv().data
        self.pg.patches_[:safe_i,:,2] /= s.view(safe_i, 1, 1, 1)
        self._rescale_deltas(s1)
        self.pg.normalize()

    def close_loop(self, i, j, n):
        MIN_NUM_INLIERS = 30

        i_pts, i_feat = self.estimate_3d_keypoints(i)
        j_pts, j_feat = self.estimate_3d_keypoints(j)
        _, _, iz = i_pts.mT
        _, _, jz = j_pts.mT
        th = 20
        i_pts = i_pts[iz < th]
        j_pts = j_pts[jz < th]
        for key in ['keypoints', 'descriptors']:
            i_feat[key] = i_feat[key][:,iz < th]
            j_feat[key] = j_feat[key][:,jz < th]

        if i_pts.numel() < MIN_NUM_INLIERS:
            return False

        out = self.matcher({"image0": i_feat, "image1": j_feat})
        i_ind, j_ind = out["matches"][0].mT
        i_pts = i_pts[i_ind]
        j_pts = j_pts[j_ind]
        assert i_pts.shape == j_pts.shape, (i_pts.shape, j_pts.shape)
        i_pts, j_pts = asnumpy(i_pts.double()), asnumpy(j_pts.double())

        if i_pts.size < MIN_NUM_INLIERS:
            return False

        r, t, s, num_inliers = ransac_umeyama(i_pts, j_pts, iterations=400, threshold=0.1)

        if num_inliers < MIN_NUM_INLIERS:
            return False

        far_rel_pose = make_pypose_Sim3(r, t, s)[None]
        Gi = pp.SE3(self.pg.poses[:,self.loop_ii])
        Gj = pp.SE3(self.pg.poses[:,self.loop_jj])
        Gij = Gj * Gi.Inv()
        prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu()
        loop_poses = pp.Sim3(torch.cat((prev_sim3, far_rel_pose)))
        loop_ii = torch.cat((self.loop_ii, torch.tensor([i])))
        loop_jj = torch.cat((self.loop_jj, torch.tensor([j])))

        pred_poses = pp.SE3(self.pg.poses_[:n]).Inv().cpu()

        self.loop_ii = loop_ii
        self.loop_jj = loop_jj

        torch.set_num_threads(1)

        self.lc_in_progress = True
        self.lc_process = self.lc_pool.apply_async(run_DPVO_PGO, (pred_poses.data, loop_poses.data, loop_ii, loop_jj, self.result_queue))
        return True


# ---------------------------------------------------------------------------
# Umeyama alignment (numba)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def umeyama_alignment(x, y):
    m, n = x.shape

    mean_x = x.sum(axis=1) / n
    mean_y = y.sum(axis=1) / n

    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        return None, None, None

    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1

    r = u.dot(s).dot(v)

    c = 1 / sigma_x * np.trace(np.diag(d).dot(s))
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c

@nb.njit(cache=True)
def ransac_umeyama(src_points, dst_points, iterations=1, threshold=0.1):
    best_inliers = 0
    best_R = None
    best_t = None
    best_s = None
    for _ in range(iterations):
        indices = np.random.choice(src_points.shape[0], 3, replace=False)
        src_sample = src_points[indices]
        dst_sample = dst_points[indices]

        R, t, s = umeyama_alignment(src_sample.T, dst_sample.T)
        if t is None:
            continue

        transformed = (src_points @ (R * s).T) + t

        distances = np.sum((transformed - dst_points)**2, axis=1)**0.5
        inlier_mask = distances < threshold
        inliers = np.sum(inlier_mask)

        if inliers > best_inliers:
            best_R, best_t, best_s = umeyama_alignment(src_points[inlier_mask].T, dst_points[inlier_mask].T)

        if inliers > 100:
            break

    return best_R, best_t, best_s, best_inliers
