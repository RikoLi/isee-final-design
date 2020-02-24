'''
Definitions of shape detector class and implementation.
'''
import cv2.cv2 as cv
import numpy as np

class ShapeDetector:
    '''
    Shape detector class.
    '''
    def getContourNumber(self, contour):
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

    def getPosForCluster(self, img):
        '''
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

    def getPosForMask(self, img):
        '''
        Get coordinates of each positive pixel in an image.\n
        img: np.array\n
        return: np.array
            '''
        coords = []
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if img[i,j] == 255:
                    coords.append([j, i])
        return np.array(coords)

    def kmeansCluster(self, bin_img, k=5):
        '''
        K-means clustering to determine possible ROI.\n
        img: np.array, binary image\n
        k: int, number of cluster centers\n
        return: labels, centers
        '''
        coords = self.getPosForCluster(bin_img)
        _, labels, centers = cv.kmeans(coords, k, None, (cv.TERM_CRITERIA_EPS, 0, 0.1), 1, cv.KMEANS_RANDOM_CENTERS)
        return labels, centers

    def cropByHue(self, img):
        mask = cv.inRange(img, 150, 180) # Crop for red zone in hue
        dst = cv.bitwise_and(mask, img)
        dst = cv.GaussianBlur(dst, (5,5), 10)
        _, dst = cv.threshold(dst, 0, 255, cv.THRESH_OTSU)
        return dst
    
    def detectSwitchROI(self, img):
        '''
        Detect switch ROI in an image.\n
        :param img: np.array, input image\n
        :return: np.array, list, ROI with bounding box and coordinates of possible ROI centers
        '''
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        v = cv.bilateralFilter(v, 0, 15, 15)
        _, v = cv.threshold(v, 0, 255, cv.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        v = cv.morphologyEx(v, cv.MORPH_ERODE, kernel)
        points = self.getPosForMask(v)
        xs, ys, dx, dy = cv.boundingRect(points)
        roi = h[ys:ys+dy, xs:xs+dx]
        crop = self.cropByHue(roi)
        kernel = np.ones((7, 7), np.uint8)
        crop = cv.morphologyEx(crop, cv.MORPH_CLOSE, kernel)
        crop = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel)
        _, contours, _ = cv.findContours(crop, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        centers = []

        for i in range(len(contours)):
            # img = cv.drawContours(img, contours, i, (0,0,255), 1)
            points = [x[0] for x in contours[i]]
            xs, ys, dx, dy = cv.boundingRect(np.array(points))
            img = cv.rectangle(img, (xs, ys), (xs+dx, ys+dy), (0, 255, 0), 1)
            center = [xs + int(dx/2), ys + int(dy/2)]
            img = cv.circle(img, tuple(center), 3, (0, 255, 0), thickness=cv.FILLED)
            centers.append(center)
        return img, centers

# for test
if __name__ == "__main__":
    # img = cv.imread('../../images/IMG_7966.JPG', 1)
    img = cv.imread('../../images/IMG_7958.JPG', 1)
    # img = cv.imread('../../images/IMG_7967.JPG', 1)
    rows = img.shape[0]#1024
    cols = img.shape[1]#768
    new_size = (cols, rows)
    img = cv.resize(img, new_size)
    img = img[int(rows/3):int(rows/3*2)+1, :int(cols/4)]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    p = ShapeDetector()
    out, centers = p.detectSwitchROI(img)
    print(centers)
    cv.imwrite('result2_original_size.png', img)
