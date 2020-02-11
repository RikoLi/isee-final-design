'''
Definitions of shape detector class and implementation.
'''
import cv2.cv2 as cv
import numpy as np

class ShapeDetector:
    '''
    Shape detector class.
    '''
    def _getContourNumber(self, contour):
        '''
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

    def _kmeansCluster(self, bin_img, k=5):
        '''
        K-means clustering to determine possible ROI.\n
        img: np.array, binary image\n
        k: int, number of cluster centers\n
        return: labels, centers
        '''
        coords = self._getPosForCluster(bin_img)
        _, labels, centers = cv.kmeans(coords, k, None, (cv.TERM_CRITERIA_EPS, 0, 0.1), 1, cv.KMEANS_RANDOM_CENTERS)
        return labels, centers

    def _cropByHue(self, img):
        mask = cv.inRange(img, 150, 180) # Crop for red zone in hue
        dst = cv.bitwise_and(mask, img)
        dst = cv.GaussianBlur(dst, (5,5), 10)
        _, dst = cv.threshold(dst, 0, 255, cv.THRESH_OTSU)
        return dst
    
    def detect(self, shape_code):
        pass

# for test
if __name__ == "__main__":
    # img = cv.imread('../../images/IMG_7966.JPG', 1)
    # img = cv.imread('../../images/IMG_7958.JPG', 1)
    img = cv.imread('../../images/IMG_7967.JPG', 1)
    img = cv.resize(img, (600, 800))
    # img = img[200:601, :150]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # cv.imshow('hue', h)
    # _, h = cv.threshold(h, 0, 255, cv.THRESH_OTSU)
    # kernel = np.ones((3,3), np.uint8)
    # h = cv.morphologyEx(h, cv.MORPH_CLOSE, kernel)
    # h = cv.morphologyEx(h, cv.MORPH_OPEN, kernel)

    p = ShapeDetector()
    # labels, centers = p._kmeansCluster(h, k=50)
    crop = p._cropByHue(h)
    kernel = np.ones((7,7), np.uint8)
    crop = cv.morphologyEx(crop, cv.MORPH_CLOSE, kernel)
    crop = cv.morphologyEx(crop, cv.MORPH_OPEN, kernel)

    # for cent in centers:
    #     cent = tuple([int(x) for x in cent])
    #     img = cv.circle(img, tuple(cent), 3, (0,255,0), cv.FILLED)

    # cv.imshow('img', img)
    cv.imshow('thres', h)
    cv.imshow('crop', crop)
    cv.waitKey()
    cv.destroyAllWindows()