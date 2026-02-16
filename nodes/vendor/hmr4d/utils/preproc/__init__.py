import logging

# Tracker not used in ComfyUI (masks replace YOLO tracking)
# from hmr4d.utils.preproc.tracker import Tracker
from ...utils.preproc.vitfeat_extractor import Extractor
from ...utils.preproc.vitpose import VitPoseExtractor
from ...utils.preproc.relpose.simple_vo import SimpleVO

log = logging.getLogger("motioncapture")

try:
    from ...utils.preproc.slam import SLAMModel
except Exception as e:
    log.debug("SLAMModel not available (DPVO not installed): %s", e)
    pass
