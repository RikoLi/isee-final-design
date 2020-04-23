'''
Definition of tooth detector class.
'''
import os
import itertools
import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt
from utils import get_points
from scipy.ndimage.filters import gaussian_filter1d, maximum_filter1d

class ToothDetector:
    '''
    Create a tooth detector.
    '''
    def __init__(self):
        self.radius = 0

    def _polar_warp_and_rotate(self, img, ellipse):
        '''
        Take the center of a valve as origin and warp it into x-y plane.

        @img:
            np.array, input valve ROI.
        @ellipse:
            tuple, it should be the output of the valve detector.
        @return:
            np.array, image after polar warping and 90-degree counter-clockwise rotation.
        '''
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        self.radius = int(max(ellipse[1])/2)
        mask = np.zeros(img.shape, dtype=np.uint8)
        mask = cv.circle(mask, center, self.radius, (255,255,255), cv.FILLED)
        img = cv.bitwise_and(mask, img)
        img = cv.warpPolar(img, (0,0), center, max(img.shape), cv.WARP_POLAR_LINEAR+cv.WARP_FILL_OUTLIERS)
        img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _preprocess(self, warped_img):
        '''
        Preprocess the warped and rotated image.

        @warped_img:
            np.array, it should be the output of self._polar_warp_and_rotate().
        @return:
            (s_mask, output_img), saturation mask and image after preprocessing.
        '''
        warped_img = cv.GaussianBlur(warped_img, (3,3), 1.5)
        hsv = cv.cvtColor(warped_img, cv.COLOR_BGR2HSV)
        warped_img = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
        warped_img = cv.equalizeHist(warped_img) # Enhance contrast

        _, s, _ = cv.split(hsv)
        _, s = cv.threshold(s, 0, 255, cv.THRESH_OTSU)
        s = cv.morphologyEx(s, cv.MORPH_ERODE, np.ones((5,5)))
        _, contours, _ = cv.findContours(s, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda ctr: cv.contourArea(ctr)) # Sort to choose the largest area
        mask = cv.drawContours(np.zeros((warped_img.shape), np.uint8), contours, len(contours)-1, (255,255,255), thickness=1)
        box = cv.boundingRect(get_points(mask)) # Largest area box-bouding
        mask = cv.rectangle(mask, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,255,255), cv.FILLED) # Fill the area that is to be removed
        mask = cv.bitwise_not(mask) # Ensure tooth existing area
        return mask, warped_img

    def _get_gradient_x(self, mask, warped_img):
        '''
        Compute the gradient of masked area using Sobel operator in x direction.

        @mask:
            np.array, it should be the output of self._preprocess().
        @warped_img:
            np.array, it should be the output of self._preprocess().
        @return:
            np.array, gradient image of the masked area in the warped image.
        '''
        dx = cv.Sobel(warped_img, 0, 1, 0)
        dx = np.abs(dx)
        dx = cv.bitwise_and(mask, dx)
        return dx

    def _accumulate_gradient(self, gradient):
        '''
        Accumulate gradient values in y direction.

        @gradient:
            np.array, input gradient image.
        @return:
            np.array, accumulated gradient values in y direction after smoothing. It starts from the first column to the last one.
        '''
        rows, cols = gradient.shape
        col_accum = []
        for j in range(cols):
            s = 0
            for i in range(rows):
                s += gradient[i][j]
            col_accum.append(s)
        col_accum = gaussian_filter1d(np.array(col_accum), 1)
        return col_accum
    def _gradient_nms(self, accumulated_grad, win_size=3):
        '''
        Non-maximum suppression to find local maximum of accumulated gradient.

        @accumulated_grad:
            np.array, input accumulated gradient.
        @win_size:
            int, sliding window length. It decides how much you refer to neighbor field values.
        @return:
            list, local maximum accumulated gradient indexs.
        '''
        indexs = []
        maximas = maximum_filter1d(accumulated_grad, win_size)
        for i in range(len(accumulated_grad)):
            if accumulated_grad[i] > 0 and abs(accumulated_grad[i] - maximas[i]) < 1e-6:
                indexs.append(i)
        return indexs

    def _visualize_accumulated_gradient(self, accumulated_grad, is_save=False):
        '''
        (Debug) Visualize accumulated gradient.

        @accumulated_grad:
            np.array, input accumulated gradient.
        '''
        mean = np.mean(accumulated_grad)
        std = np.std(accumulated_grad)
        x = np.linspace(0, len(accumulated_grad)-1, len(accumulated_grad))
        plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)
        plt.plot(x, accumulated_grad)
        plt.fill_between(x, 0, accumulated_grad, color='g', alpha=0.5)
        plt.plot(x, mean*np.ones((len(accumulated_grad),)), color='r', linestyle='-.', label='mean')
        plt.plot(x, (mean+std)*np.ones((len(accumulated_grad),)), color='pink', linestyle='-.', label='mean+std')
        plt.plot(x, (mean+2*std)*np.ones((len(accumulated_grad),)), color='purple', linestyle='-.', label='mean+2*std')
        plt.legend()
        plt.xlabel('Column')
        plt.ylabel('Accumulated Gradient')
        plt.title('Distribution of Accumulated Gradient on Columns')
        if is_save:
            plt.savefig('accumulated_gradient.png', dpi=600)
            print('Plot is saved.')
        plt.show()

    def _check_distance(self, accumulated_grad, col_id, standard_dist, eps=10):
        '''
        Check and choose legal column index pairs.

        @col_id:
            list, column indexs.
        @standard_dist:
            float, standard distance between two teeth, it should be estimated previously.
        @eps:
            float, acceptable error.
        @return:
            list, legal column index pairs.
        '''
        comb = itertools.combinations(col_id, 2)
        legal_pair = []
        for pair in comb:
            dist = abs(pair[0] - pair[1])
            print('dist:{0:.1f} standard:{1:.1f} diff:{2:.1f}'.format(dist, standard_dist, abs(dist-standard_dist))) # debug
            if abs(dist - standard_dist) < eps:
                legal_pair.append(pair)
        return legal_pair
    
    def detect(self, img, ellipse, draw_results=False):
        '''
        Detect teeth in given valve image.

        @img:
            np.array, input valve ROI.
        @ellipse:
            tuple, it should be the output of the valve detector.
        @return:
            (list, warped), coordinates of teeth and warped image.
        '''
        warped = self._polar_warp_and_rotate(img, ellipse)
        mask, warped_new = self._preprocess(warped)
        grad = self._get_gradient_x(mask, warped_new)
        accum = self._accumulate_gradient(grad)

        self._visualize_accumulated_gradient(accum) # debug

        mean = np.mean(accum)
        std = np.std(accum)
        thresh = mean + 2 * std # Set threshold to find "extreme points"
        for i, ac in enumerate(accum):
            if ac < thresh:
                accum[i] = 0
        possible_id = self._gradient_nms(accum, 3)
        legal_pairs = self._check_distance(accum, possible_id, warped.shape[1]/2, eps=15)
        if draw_results:
            for idx in possible_id:
                warped = cv.line(warped, (idx,0), (idx,warped.shape[0]-1), (0,255,0), 1)
            for pair in legal_pairs:
                warped = cv.line(warped, (pair[0],0), (pair[0],warped.shape[0]-1), (0,0,255), 1)
                warped = cv.line(warped, (pair[1],0), (pair[1],warped.shape[0]-1), (0,0,255), 1)
        return possible_id, warped

if __name__ == "__main__":
    from valve_detection import ValveDetector
    d = ValveDetector()
    new_scale = (100, 100)
    td = ToothDetector()
    PREFIX = '../../rois'
    imgs = ['roi_124_valve_0', 'roi_15_valve_0', 'roi_27_valve_0',
            'roi_71_valve_0', 'roi_74_valve_0', 'roi_82_valve_0',
            'roi_88_valve_0', 'roi_99_valve_0', 'roi_108_valve_0',
            'roi_115_valve_0', 'roi_128_valve_0', 'roi_129_valve_0',
            'roi_133_valve_0', 'roi_157_valve_0', 'roi_159_valve_0',
            'roi_165_valve_0', 'roi_188_valve_0', 'roi_192_valve_0',
            'roi_201_valve_0', 'roi_206_valve_0', 'roi_212_valve_0']
    imgs = [PREFIX + '/' + img + '.png' for img in imgs]
    for i in range(len(imgs)):
        img = cv.imread(imgs[i+5], 1)
        img = d.down_scale(img, new_scale)
        _, elps = d.detect_ransac(img)
        teeth, warped = td.detect(img, elps, draw_results=True)
        cv.imshow('warped', warped)
        cv.imshow('image', img)
        key = cv.waitKey()
        if key == ord('n'):
            cv.destroyAllWindows()
        else:
            quit()
