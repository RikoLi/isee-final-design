'''
Definitions of switch detector class and implementation.
'''
import cv2.cv2 as cv
import numpy as np

class SwitchDetector:
    '''
    Create a switch detector.\n
    @min_area:
        int, minimum contour area that will be considered as a proposal ROI
    @wh_ratio:
        float, target standard width-height ratio
    @ratio_eps:
        float, maximum error of width-height ratio
    @return:
        SwitchDetector
    '''
    def __init__(self, min_area=50, wh_ratio=1.0, ratio_eps=0.6):
        self.min_area = min_area
        self.ratio_eps = ratio_eps
        self.wh_ratio = wh_ratio
        self.ratio_eps = ratio_eps

    def _getContourNumber(self, contour):
        '''
        ### Now deprecated ! ###
        Computer the contour number for a given closed contour.\n
        contour: list, list of points on the contour\n
        return: float, contour number.
        '''
        contour = contour.tolist() # Convert to list
        contour = [x[0] for x in contour]
        num = len(contour) # Numbers of points
        g_vec = [] # Container for normal vectors
        m_vec = [] # Container for mass center vectors
        a_vec = [] # Container for angles

        # Compute mass center position
        x_sum = sum([pt[0] for pt in contour])
        y_sum = sum([pt[1] for pt in contour])
        mass_center = (x_sum/num, y_sum/num) # sub-pixel-level representation

        # Compute normal vectors
        for i in range(num):
            # Firstly, compute tangent vector
            if i + 1 == num:
                tangent_v = [contour[0][0]-contour[i][0], contour[0][1]-contour[i][1]]
            else:
                tangent_v = [contour[i+1][0]-contour[i][0], contour[i+1][1]-contour[i][1]]
            # Then compute normal vector
            g = [tangent_v[1], -tangent_v[0]] # pi/2 clockwise rotation
            # Normalization
            norm = np.linalg.norm(np.array(g), 2)
            g = [sub / norm for sub in g]
            g_vec.append(g)

        # Compute mass center vectors
        for i in range(num):
            x = mass_center[0] - contour[i][0]
            y = mass_center[1] - contour[i][1]
            # Normalization
            norm = np.linalg.norm(np.array([x,y]), 2)
            x /= norm
            y /= norm
            m_vec.append([x, y])
        
        # Compute angle
        for i in range(num):
            cosine = np.dot(np.array(g_vec[i]), np.array(m_vec[i])) # denominator is 1
            a_vec.append(np.arccos(cosine) / np.pi * 180)
        return sum(a_vec) / num

    def _getPosForCluster(self, img):
        '''
        ### Now deprecated ! ###
        Get coordinates of each available pixel in an image.\n
        img: np.array\n
        return: np.array, list of coordinates
        '''
        coords = []
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if img[i,j] == 255:
                    coords.append(np.array([j, i]))
        return np.array(coords, np.float32)

    def _getPosForMask(self, img):
        '''
        Get coordinates of each positive pixel in an image.\n
        @img:
            np.array, it should be a binary, single-channel image
        @return:
            np.array, coordinates of positive pixels
        '''
        coords = []
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if img[i,j] == 255:
                    coords.append([j, i])
        return np.array(coords)

    def _kmeansCluster(self, bin_img, k=5):
        '''
        ### Now deprecated ! ###
        K-means clustering to determine possible ROI.\n
        img: np.array, binary image\n
        k: int, number of cluster centers\n
        return: labels, centers
        '''
        coords = self._getPosForCluster(bin_img)
        _, labels, centers = cv.kmeans(coords, k, None, (cv.TERM_CRITERIA_EPS, 0, 0.1), 1, cv.KMEANS_RANDOM_CENTERS)
        return labels, centers

    def _cropByHue(self, img):
        '''
        ### Now deprecated! ###
        '''
        lower_bound = 5
        upper_bound = 175
        inv_mask = cv.inRange(img, lower_bound, upper_bound)
        red_mask = cv.bitwise_not(inv_mask) # Crop for red zone in hue
        dst = cv.bitwise_and(red_mask, img)
        dst = cv.GaussianBlur(dst, (5,5), 10)
        _, dst = cv.threshold(dst, 0, 255, cv.THRESH_OTSU)
        return dst

    def detectSwitchROI(self, img,\
        lower_bound=np.array([110, 55, 5]),\
        upper_bound=np.array([130, 255, 255])):
        '''
        Detect switch ROI in an image.\n
        @img:
            np.array, input image
        @lower_bound:
            np.array, lower bound for HSV color cut
        @upper_bound:
            np.array, upper bound for HSV color cut
        @return:
            np.array, list, ROI image with bounding box and ROI bounding boxes, each is a tuple like (x, y, dx, dy)
        '''
        # Replace R and B channel
        blurred_img = cv.GaussianBlur(img, (5,5), 5) # Gaussian smoothing to remove noisy points
        b,g,r = cv.split(blurred_img)
        inv_img = cv.merge([r,g,b])
        hsv = cv.cvtColor(inv_img, cv.COLOR_BGR2HSV)

        # Crop for color mask
        mask = cv.inRange(hsv, lower_bound, upper_bound)

        # Crop for ROI
        h, s, v = cv.split(hsv)
        v = cv.bilateralFilter(v, 0, 15, 15)
        _, v = cv.threshold(v, 0, 255, cv.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        v = cv.morphologyEx(v, cv.MORPH_ERODE, kernel)
        # cv.imshow('value', v)
        points = self._getPosForMask(v)
        xs, ys, dx, dy = cv.boundingRect(points)
        # img = cv.rectangle(img, (xs, ys), (xs+dx, ys+dy), (255,0,0), 2)
        roi_mask = mask[ys:ys+dy, xs:xs+dx]
        kernel = np.ones((3, 3), np.uint8)
        roi_mask = cv.morphologyEx(roi_mask, cv.MORPH_CLOSE, kernel) # Trimming
        roi = img[ys:ys+dy, xs:xs+dx]


        _, contours, _ = cv.findContours(roi_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # centers = []
        boxes = []

        for i in range(len(contours)):
            # roi = cv.drawContours(roi, contours, i, (255,0,0), 1)
            points = [x[0] for x in contours[i]]
            area = cv.contourArea(contours[i]) # filter out small areas
            if area < self.min_area:
                continue
            xs, ys, dx, dy = cv.boundingRect(np.array(points))
            # centers.append(center)
            boxes.append((xs, ys, dx, dy))
        
        # debug
        # cv.imshow('mask', mask)
        # cv.imshow('roi', roi)
        # cv.imshow('roi_mask', roi_mask)
        return roi, boxes

    def _spacialDiff(self, img, dx=1, dy=1):
        '''
        ### Now deprecated ! ###
        Get spacial difference image with offset dx and dy.
        '''
        H = np.zeros((2, 3))
        H[0, 0] = 1
        H[0, 2] = dx
        H[1, 1] = 1
        H[1, 2] = dy
        translated = cv.warpAffine(img, H, (img.shape[1], img.shape[0]))
        img = translated - img
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        return np.uint8(img)

    def _checkBoxRatio(self, width_height):
        '''
        Check whether a bounding box satisties a given width-height ratio.\n
        @width_height:
            tuple, bounding box parameters, like (w, h), that means the width and height of the box
        @return:
            bool, whether the box satiesfies the standard or not
        '''
        is_normal = False
        w, h = width_height
        if abs(w / h - self.wh_ratio) < self.ratio_eps:
            is_normal = True
        return is_normal

    def checkBoxes(self, boxes):
        '''
        Check a list of bounding boxes to pick out the best-matched bounding box of ROI.\n
        @boxes:
            list, list of bounding boxes, like [(x,y,dx,dy), ...]
        @return:
            list, the best-matched bounding boxes of ROI, each is a tuple like (x, y, dx, dy)
        '''
        realBoxes = None

        # Width-height ratio filtering
        realBoxes = [box for box in boxes if self._checkBoxRatio(box[2:])]

        # maybe more ...
        return realBoxes

    def drawBoundingBox(self, img, boxes):
        '''
        Draw all bounding boxes on the given image.\n
        @img:
            np.array, input image
        @boxes:
            list, list of bounding boxes, each is a tuple like (x, y, dx, dy)
        @return:
            np.array, image with bounding boxes
        '''
        for box in boxes:
            xs, ys, dx, dy = box
            img = cv.rectangle(img, (xs, ys), (xs+dx, ys+dy), (0, 255, 0), 1)
            center = [xs + int(dx/2), ys + int(dy/2)]
            img = cv.circle(img, tuple(center), 2, (0, 255, 0), thickness=cv.FILLED)
        return img

# for test
if __name__ == "__main__":
    # img = cv.imread('../../images/fixed/2.26_fixed_box_left.png', 1)
    # img = cv.imread('../../images/fixed/2.26_fixed_box_leftup.png', 1)
    # img = cv.imread('../../images/fixed/2.26_fixed_box_right.png', 1)
    # img = cv.imread('../../images/fixed/2.26_fixed_box_right2.png', 1)
    # img = cv.imread('../../images/original/IMG_7966.JPG', 1)
    # img = cv.imread('../../images/original/IMG_7958.JPG', 1)
    img = cv.imread('../../images/original/3.8_box1.jpg', 1)
    # img = cv.imread('../../images/original/3.8_box2.jpg', 1)
    # img = cv.imread('../../images/original/3.8_box3.jpg', 1)
    # img = cv.imread('../../images/original/3.8_box4.jpg', 1)
    
    rows = 1024#1024
    cols = 768#768
    new_size = (cols, rows)
    img = cv.resize(img, new_size)
    img = img[int(rows/3):int(rows/3*2)+1, :int(cols/4)]
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    p = SwitchDetector()
    roi, boxes = p.detectSwitchROI(img)
    boxes = p.checkBoxes(boxes)
    out = p.drawBoundingBox(roi, boxes)

    cv.imshow('out', out)
    cv.waitKey()
    cv.destroyAllWindows()
    # cv.imwrite('3.3_detect_box_right2.png', out)