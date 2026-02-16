import torch
import pytorch_lightning as pl
from ...utils.pylogger import Log

from ...utils.geo.hmr_cam import normalize_kp2d


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    @torch.no_grad()
    def predict(self, data, static_cam=False):
        """auto add batch dim
        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        """
        # ROPE inference
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        batch = {k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device) for k, v in batch.items()}
        outputs = self.pipeline.forward(batch, train=False, postproc=True, static_cam=static_cam)

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # intermediate outputs
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        """Load pretrained checkpoint, and assign each weight to the corresponding part."""
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        import comfy.utils
        state_dict = comfy.utils.load_torch_file(str(ckpt_path))
        missing, unexpected = self.load_state_dict(state_dict, strict=False, assign=True)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")
