import yaml


class DPVOConfig:
    """DPVO configuration -- replaces yacs CfgNode."""

    # max number of keyframes
    BUFFER_SIZE = 4096

    # bias patch selection towards high gradient regions?
    CENTROID_SEL_STRAT = 'RANDOM'

    # VO config (increase for better accuracy)
    PATCHES_PER_FRAME = 80
    REMOVAL_WINDOW = 20
    OPTIMIZATION_WINDOW = 12
    PATCH_LIFETIME = 12

    # threshold for keyframe removal
    KEYFRAME_INDEX = 4
    KEYFRAME_THRESH = 12.5

    # camera motion model
    MOTION_MODEL = 'DAMPED_LINEAR'
    MOTION_DAMPING = 0.5

    MIXED_PRECISION = True

    # Loop closure
    LOOP_CLOSURE = False
    BACKEND_THRESH = 64.0
    MAX_EDGE_AGE = 1000
    GLOBAL_OPT_FREQ = 15

    # Classic loop closure
    CLASSIC_LOOP_CLOSURE = False
    LOOP_CLOSE_WINDOW_SIZE = 3
    LOOP_RETR_THRESH = 0.04

    def merge_from_file(self, path):
        with open(path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            for k, v in overrides.items():
                setattr(self, k, v)


cfg = DPVOConfig()
