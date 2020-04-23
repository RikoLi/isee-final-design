'''
Definition of valve detector class and implementation.
'''
import numpy as np
import cv2.cv2 as cv
import os

class ValveDetector:
    '''
    Create a valve detector.

    @kernel_size:
        tuple, kernel size for gaussian smoothing.
    @sigma:
        float, standard deviation for gaussian smoothing.
    @canny_param:
        tuple, canny edge extraction parameters.
    @downscale_size:
        tuple, target down-scaled image size.
    '''
    def __init__(
        self,
        kernel_size=(5,5),
        sigma=1.0,
        canny_param=(150,200),
        downscale_size=(100,100)
    ):
        self._gassian_kernel_size = kernel_size
        self._gassian_sigma = sigma
        self._canny_param = canny_param
        self._scale_factor = None
        self._downscale_size = downscale_size

    def down_scale(self, img, target_size):
        '''
        Down-scale given image to target size.

        @img:
            np.array, input image.
        @target_size:
            tuple, target image size. It should be (x, y).
        @return:
            np.array, down-scaled image.
        '''
        scaled = cv.resize(img, self._downscale_size)
        self._scale_factor = (img.shape[1]/target_size[0], img.shape[0]/target_size[1])
        return scaled

    def recover_scale(self, img):
        '''
        Recover original image size.

        @img:
            np.array, input image.
        @return:
            (img, scale_factor), image in original size and scale factors.
        '''
        img = cv.resize(img, (0,0), fx=self._scale_factor[0], fy=self._scale_factor[1], interpolation=cv.INTER_AREA)
        return img, self._scale_factor

    def recover_ellipse(self, param, scale_factor):
        fx, fy = scale_factor
        center, axis, angle = param
        x0, y0 = center
        short, long = axis
        x_long_top = x0 + np.cos(angle) * long / 2
        y_long_top = y0 + np.sin(angle) * long / 2
        x_short_top = x0 - np.sin(angle) * short / 2
        y_short_top = y0 + np.cos(angle) * short / 2
        long_new = np.sqrt(fx**2 * (x0-x_long_top)**2 + fy**2 * (y0-y_long_top)**2) * 2
        short_new = np.sqrt(fx**2 * (x0-x_short_top)**2 + fy**2 * (y0-y_short_top)**2) * 2
        if long_new < short_new:
            angle = angle + 90
        return ((x0 * fx, y0 * fy), (min(long_new, short_new), max(long_new, short_new)), angle)

    def _draw_circles(self, img, circles):
        circles = circles[0]
        for c in circles:
            img = cv.circle(img, (c[0], c[1]), c[2], (0,255,0), 1)
        cv.imshow('circles', img)

    def _pixel_sigmoid(self, x):
        '''### Now deprecated! ###'''
        x = x - 255.0 / 2 # Translation
        y = 255.0 / (1 + np.exp(-x)) # Scaling
        return y.astype(np.uint8)

    def _gray_augment(self, gray):
        out = self._pixel_sigmoid(gray)
        return out

    def _get_points(self, binary):
        '''
        Get coordinates of each positive pixel in a binary image.

        @binary:
            np.array, input binary image.
        @return:
            np.array, list of coordinates.
        '''
        points = []
        for i in range(binary.shape[0]):
            for j in range(binary.shape[1]):
                if binary[i, j] != 0:
                    points.append([j, i])
        return np.array(points)

    def _symmetric_loss(self, img, box):
        try:
            x, y, dx, dy = [int(x) for x in box]
            if dx % 2 != 0: # odd-length border process
                dx -= 1
                dy -= 1

            # left-up, right-up, left-down, right-down
            rois = [img[y:y+dy//2, x:x+dx//2], img[y:y+dy//2, x+dx//2:x+dx], img[y+dy//2:y+dy, x:x+dx//2], img[y+dy//2:y+dy, x+dx//2:x+dx]]
            for i in range(len(rois)):
                rois[i] = cv.cvtColor(rois[i], cv.COLOR_BGR2GRAY)
                # normalization
                rois[i] = (rois[i] / 255.0).astype(np.float64)

            # check 0 and 3 area
            loss03 = np.linalg.norm(cv.rotate(rois[0], cv.ROTATE_180).reshape(1,-1)-rois[3].reshape(1,-1))

            # check 1 and 2 area
            loss12 = np.linalg.norm(cv.rotate(rois[1], cv.ROTATE_180).reshape(1,-1)-rois[2].reshape(1,-1))

            return 0.5 * (loss12 + loss03)
        except:
            return -1

    def _random_select_points(self, points, rate):
        '''
        Randomly select points from given points with give selection rate.

        @points:
            np.array, total points.
        @rate:
            float, selection rate.
        @return:
            np.array, coordinates of selected points.
        '''
        points = points.tolist()
        selected_idxs = np.random.random_integers(0, len(points)-1, int(rate*len(points))).tolist()
        selected_points = [points[idx] for idx in selected_idxs]
        return np.array(selected_points, dtype=np.int)
    
    def _get_mean_ellipse(self, ellipses):
        '''### Now deprecated ! ###'''
        p1s = []
        p2s = []
        alphas = []
        for elps in ellipses:
            p1, p2, alpha = elps
            p1s.append(p1)
            p2s.append(p2)
            alphas.append(alpha)
        p1_mean = np.mean(np.array(p1s), axis=0, dtype=np.int64)
        p2_mean = np.mean(np.array(p2s), axis=0, dtype=np.int64)
        alpha_mean = sum(alphas) / len(ellipses)
        best = ((p1_mean[0], p1_mean[1]), (p2_mean[0], p2_mean[1]), alpha_mean)
        return best
    
    def _get_ellipse_params(self, ellipse):
        '''
        Compute general ellipse parameters using center, axis length and rotation angle.

        @ellipse:
            tuple, in such structure ((x0, y0), (2b, 2a), angle)
        @return:
            tuple, general parameters of a ellipse.
            Structure: (A, B, C, f, x0, y0),
            where the ellipse fits the general equation: A(x-x0)^2 + B(x-x0)(y-y0) + C(y-y0)^2 + f = 0
        '''
        x = ellipse[0][0]
        y = ellipse[0][1]
        a = ellipse[1][1] / 2
        b = ellipse[1][0] / 2
        theta = ellipse[-1]

        A = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
        B = 2 * (a**2 - b**2) * np.sin(theta) * np.cos(theta)
        C = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
        f = -1 * a**2 * b**2

        return A, B, C, f, x, y

    def _get_ellipse_loss(self, params, points):
        '''
        Compute direct loss of points with a fitted ellipse.

        @params:
            tuple, general parameters of an ellipse.
        @points:
            np.array, all given points used to fit an ellipse.
        @return:
            list, list of loss of all given points in the same order.
            Loss is computed directly: loss = A(x-x0)^2 + B(x-x0)(y-y0) + C(y-y0)^2 + f
        '''
        A, B, C, f, x0, y0 = params
        loss = []
        for p in points:
            x, y = p
            loss_p = A * (x-x0)**2 + B * (x-x0) * (y-y0) + C * (y-y0)**2 + f
            loss.append(loss_p)
        return loss
    
    def _choose_new_inliers(self, loss, threshold=1e6):
        '''
        Choose new inliers in each iteration.

        @loss:
            list, loss of all given points.
        @threshold:
            float, threshold which decides whether a point is an inlier.
        @return:
            list, list of indexs of chosen inliers.
        '''
        new_inliers_id = []
        for i, l in enumerate(loss):
            if abs(l) < threshold:
                new_inliers_id.append(i)
        return new_inliers_id

    def _merge_inliers_id(self, id1, id2):
        '''### Now deprecated ! ###'''
        id1 = set(id1)
        return id1.union(set(id2))
    
    def _fit_ellipse_ransac(self, points, outlier_rate=0.2, iterations=1000):
        '''
        Ellipse fit using RANSAC algorithm.

        @points:
            np.array, coordinates of points that are used to fit an ellipse.
        @outlier_rate:
            float, see ValveDetector.detect().
        @iterations:
            int, see ValveDetector.detect().
        @return:
            tuple, best fitted ellipse paramters.
        '''
        final_inliers_id = set()
        for i in range(iterations):
            new_selected_points = self._random_select_points(
                points, 1-outlier_rate)
            new_ellipse = cv.fitEllipse(new_selected_points)
            new_elps_params = self._get_ellipse_params(new_ellipse)
            points_loss = self._get_ellipse_loss(new_elps_params, points)
            new_inliers_id = self._choose_new_inliers(points_loss, threshold=2000.0)
            final_inliers_id.update(new_inliers_id)
        final_points = np.array([points[i].astype(np.int) for i in final_inliers_id])
        best_ellipse = cv.fitEllipse(final_points)
        return best_ellipse

    def _get_saturation_edge(self, img):
        '''
        Get image edges for saturation channel in HSV model.

        @img:
            np.array, input image.
        @return:
            np.array, edge image.
        '''
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        _, s, _ = cv.split(hsv)
        s = cv.equalizeHist(s)
        s = cv.GaussianBlur(s, self._gassian_kernel_size, self._gassian_sigma)
        s = cv.Canny(s, max(self._canny_param), min(self._canny_param))
        return s

    def detect_ransac(self, img, outlier_rate=0.4, iterations=1000, draw_ellipse=False):
        '''
        Detect valve center with RANSAC ellipse fitting algorithm.

        @img:
            np.array, input image.
        @outlier_rate
            float, rate of pre-estimated outliers in fitting points, should be larger than 0 and smaller than 1.0.
        @iterations
            int, positive int, which means the total iterations of RANSAC algorithm.
        @return
            (np.array, ellipse), image with fitted ellipse and ellipse parameters.
            Parameters are exactly the same as the return values of OpenCV fitEllipse() function.
        '''
        edge = self._get_saturation_edge(img)
        points = self._get_points(edge)
        
        # traditional fitting
        # elps = cv.fitEllipse(points)
        # center, _, _ = elps
        # x = int(center[0])
        # y = int(center[1])
        # img = cv.ellipse(img, elps, color=(0,255,0), thickness=1)
        # img = cv.circle(img, (x, y), 2, (0,255,0), cv.FILLED)

        # ransac
        ellipse = self._fit_ellipse_ransac(points, outlier_rate, iterations)
        center, params, _ = ellipse
        x = int(center[0])
        y = int(center[1])
        if draw_ellipse:
            img = cv.ellipse(img, ellipse, color=(0,255,255), thickness=1)
            img = cv.circle(img, (x, y), 2, (0,255,255), cv.FILLED)
        # print('ratio of axis(short/long):{0:.3f}'.format(params[0]/params[1]))
        return img, ellipse

if __name__ == "__main__":
    img_folder = '../../rois'
    img_names = os.listdir(img_folder)
    img_paths = [os.path.join(img_folder, name) for name in img_names]
    for i in range(len(img_paths)):
        img = cv.imread(img_paths[i+2], 1)
        d = ValveDetector()
        scaled = d.down_scale(img, (100,100))
        # img = cv.resize(img, (100,100))
        scaled, param = d.detect_ransac(scaled, outlier_rate=0.4, iterations=1000)
        recovered, factors = d.recover_scale(scaled)
        # param = d.recover_ellipse(param, factors)
        cv.imshow('result', scaled)
        cv.imshow('raw', img)
        cv.imshow('recovered', recovered)
        key = cv.waitKey()
        if key == ord('n'):
            pass
        elif key == ord('s'):
            cv.imwrite('test_{}.png'.format(i), recovered)
            print('image saved')
        else:
            quit()
        cv.destroyAllWindows()