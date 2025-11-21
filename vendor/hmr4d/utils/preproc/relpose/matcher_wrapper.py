from .model.base_matcher import BaseMatcher
from .model.cv2_matcher import CV2SIFTMather, CV2ORBMather


matcher_map = {
    "sift": CV2SIFTMather,
    "orb": CV2ORBMather,
}


class Matcher:
    def __init__(self, matcher="sift", args=None):
        self.matcher: BaseMatcher = matcher_map[matcher](args)

    def match_np(self, img0, img1):
        """
        Args:
            img0: np.ndarray, shape (H, W, 3), dtype=np.uint8
            img1: np.ndarray, shape (H, W, 3), dtype=np.uint8
        Returns:
            pts0: np.ndarray, shape (N, 2), dtype=np.float32
            pts1: np.ndarray, shape (N, 2), dtype=np.float32
        """
        return self.matcher.match_np(img0, img1)
