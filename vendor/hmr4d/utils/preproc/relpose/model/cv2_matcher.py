import cv2
import numpy as np
from .base_matcher import BaseMatcher


class CV2SIFTMather(BaseMatcher):
    def __init__(self, args=None):
        super().__init__()
        self.sift = cv2.SIFT_create()

    def match_np(self, img0, img1):
        # Convert images to grayscale
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Find keypoints and descriptors
        kp0, des0 = self.sift.detectAndCompute(gray0, None)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)

        # Match descriptors using FLANN matcher (better for SIFT)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des0, des1, k=2)

        # Store all good matches as per Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print(
                f"Warning: Only {len(good_matches)} matches found, which might not be enough for reliable pose estimation"
            )

        # Extract matched point coordinates
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return pts0, pts1


class CV2ORBMather(BaseMatcher):
    def __init__(self, args=None):
        super().__init__()
        self.orb = cv2.ORB_create()
        self.num_matches = 1024

    def match_np(self, img0, img1):
        # Convert images to grayscale
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Original ORB method
        kp0, des0 = self.orb.detectAndCompute(gray0, None)
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[: self.num_matches]

        # Ensure we have enough matches
        if len(good_matches) < 8:
            print(
                f"Warning: Only {len(good_matches)} matches found, which might not be enough for reliable pose estimation"
            )
            # Pad with more matches if available
            if len(matches) > len(good_matches):
                good_matches = matches[: min(100, len(matches))]

        # Extract matched point coordinates
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return pts0, pts1
