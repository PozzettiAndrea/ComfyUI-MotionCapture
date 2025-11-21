import numpy as np
from .utils import focal_length_from_mm
from .matcher_wrapper import Matcher
from .solver_two_view import TwoPairSolver, CameraParams, interpolate_missing_frames
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np


class SimpleVO:
    def __init__(self, video_path, scale=0.5, step=8, method="sift", f_mm=None):
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # fullframe camera focal length in mm

    def compute(self):
        # Read video
        frames = read_video_np(self.video_path, scale=self.scale)

        # Downsample frames, and interpolate missing frames
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        print(f"[SimpleVO] Choosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO:We should use different pipelines for different methods
        T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)

        # Interpolate missing frames
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        T_w2c_list = [np.eye(4)]  # cam poses are defined as T_w2c @ p_w = p_c
        prev_frame = frames[0]
        for frame_idx in tqdm(range(1, len(frames))):
            curr_frame = frames[frame_idx]

            # Match frames
            pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
            T_delta = solver.solve(pts0, pts1)  # T_delta = T_curr @ T_last^-1

            # Compute current frame's transformation matrix
            T_w2c_list.append(T_delta @ T_w2c_list[-1])

            # Current frame becomes previous frame for next iteration
            prev_frame = curr_frame

        return T_w2c_list
