'''
Definitions of preprocessing classes and implementations.
'''
import cv2.cv2 as cv
import numpy as np

class Preprocess:
    '''
    Parent preprocess class for input images.
    '''
    def gaussianFilter(self, img, win_size=(5,5), sigma=5):
        '''
        Gaussian filtering.\n
        img: np.array\n
        win_size: tuple\n
        sigma: float\n
        return: np.array
        '''
        return cv.GaussianBlur(img, win_size, sigma)
    
    def bilateralFilter(self, img, sigma=15):
        '''
        Bilateral filtering.\n
        img: np.array\n
        return: np.array
        '''
        return cv.bilateralFilter(img, 0, sigma, sigma)
    
    def getFeature(self, img, feature_type=0):
        '''
        Get features from give image.\n
        img: np.array\n
        feature_type: int, 0:ORB, 1:SIFT\n
        return: keypoints, descriptor
        '''
        img = self.gaussianFilter(img)
        if feature_type == 0:
            # ORB
            orb = cv.ORB_create()
            kp, des = orb.detectAndCompute(img, None)
            return kp, des
        elif feature_type == 1:
            # SIFT
            sift = cv.xfeatures2d.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            return kp, des
        else:
            print('Wrong feature type code! It should be 0 or 1!')
            return None
    
    def _testDrawFeatures(self, img):
        '''
        Build-in test function for feature extraction.\n
        img: np.array
        '''
        gimg = self.gaussianFilter(img)
        kp, _ = self.getFeature(gimg, 0)
        out = None
        out = cv.drawKeypoints(img, kp, out, color=(0,0,255))
        cv.imshow('test_draw_features', out)
        cv.waitKey()
        cv.destroyAllWindows()

class AngleProcess(Preprocess):
    '''
    Child class for angle preprocessing.\n
    '''
    def _match(self, kp1, kp2, des1, des2, MIN_MATCH_COUNT=10):
        '''
        Return homography H.
        '''
        FLANN_INDEX_KDTREE = 1 # Using KD tree
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks = 50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good) > MIN_MATCH_COUNT: # Compute homography
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            return H
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            return None

    def fix(self, img, standard_img):
        feature_type = 0 # Default feature: ORB
        img = self.gaussianFilter(img)
        standard_img = self.gaussianFilter(standard_img)
        kp, des = super().getFeature(img, feature_type)
        s_kp, s_des = super().getFeature(standard_img, feature_type)
        # Matching
        H = self._match(kp, s_kp, des, s_des)
        img = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv.imwrite('angle_fixed.png', img)
        print('Angle fixed.')

class LightProcess(Preprocess):
    '''
    Child class for light preprocessing.
    '''
    def _equalizeHist(self, img):
        '''
        Canonical histogram equalization.\n
        img: np.array\n
        return: np.array
        '''
        if len(img.shape) == 3:
            b, g, r = cv.split(img)
            b = cv.equalizeHist(b)
            g = cv.equalizeHist(g)
            r = cv.equalizeHist(r)
            img = cv.merge([b,g,r])
        elif len(img.shape) == 2:
            img = cv.equalizeHist(img)
        else:
            print('Wrong color channel numbers!')
        return img

    def _equalizeFAGC(self, img):
        '''
        Fast and Adaptive Gray-level Correction.\n
        img: np.array\n
        return: np.array
        '''
        pass

    def fix(self):
        pass

# Test code
if __name__ == "__main__":
    img1 = cv.imread('../../images/IMG_7958.JPG', cv.IMREAD_COLOR)
    img2 = cv.imread('../../images/IMG_7966.JPG', cv.IMREAD_COLOR)