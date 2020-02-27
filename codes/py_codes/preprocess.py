'''
Definitions of preprocessing classes and implementations.
'''
import cv2.cv2 as cv
import numpy as np

class Preprocess:
    '''
    Parent preprocess class for input images.
    '''
    def gaussianFilter(self, img, win_size=(3,3), sigma=5):
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
        feature_type: int, 0:ORB, 1:SIFT, 2:FAST+ORB descriptor\n
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
        elif feature_type == 2:
            # FAST corners + SIFT descriptors
            fast = cv.FastFeatureDetector_create()
            sift = cv.xfeatures2d.SIFT_create()
            kp = fast.detect(img)
            kp, des = sift.compute(img, kp)
            return kp, des
        else:
            print('Wrong feature type code!')
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

    def fix(self, img, standard_img, feature_type=0):
        # Smoothing
        gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gstd = cv.cvtColor(standard_img, cv.COLOR_BGR2GRAY)
        gimg = self.gaussianFilter(gimg, (3, 3), 5)
        gstd = self.gaussianFilter(gstd, (3, 3), 5)

        # Get features
        kp, des = super().getFeature(gimg, feature_type)
        s_kp, s_des = super().getFeature(gstd, feature_type)

        # Matching
        H = self._match(kp, s_kp, des, s_des)
        img = cv.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        print('Angle fixed.')
        return img

class LightProcess(Preprocess):
    '''
    Child class for light preprocessing.
    '''
    def homomorphicFilter(self, img, rh=1.5, rl=0.9, c=1, n=1, m=3, d0=None):
        '''
        Homomorphic filtering.\n
        img: np.array\n
        rh: float, upper bound\n
        rl: float, lower bound\n
        c: float, slope factor, rl < c < rh\n
        n: int, dynamic factor\n
        m: int, dynamic factor\n
        d0: float, cut frequency, default: 0.5 * MAX(d)\n
        return: np.array, single-channel
        '''
        eps = 1e-16
        freq = np.fft.fft2(np.log(img+eps))
        freq = np.fft.fftshift(freq)
        # Homomorphic filter design
        g = np.zeros(freq.shape)
        g = np.complex64(g) # convert to complex type
        rows, cols = freq.shape
        if d0 == None:
            d0 = 0.5 * (((rows/2)**2 + (cols/2)**2) ** 0.5)
        for x in range(cols):
            for y in range(rows):
                numerator = rh - rl
                denominator = 1 + c * (d0**n / (eps + ((y - rows/2)**2 + (x - cols/2)**2)**0.5)**m) ** 2
                h = numerator / denominator + rl
                g[y,x] = h * freq[y,x]
        g = np.fft.ifftshift(g)
        g = np.abs(np.fft.ifft2(g))
        g = np.exp(g)
        return np.uint8(g)

# Test code
if __name__ == "__main__":
    standard_img = cv.imread('../../images/original/2.26_box_standard2.JPG', 1)
    # img = cv.imread('../../images/original/2.26_box_left.JPG', 1)
    # img = cv.imread('../../images/original/2.26_box_right.JPG', 1)
    # img = cv.imread('../../images/original/2.26_box_right2.JPG', 1)
    img = cv.imread('../../images/original/2.26_box_leftup.JPG', 1)

    p = AngleProcess()
    out = p.fix(img, standard_img, 2)
    cv.imwrite('2.26_fixed_box_leftup.png', out)
    
    # cv.waitKey()
    # cv.destroyAllWindows()