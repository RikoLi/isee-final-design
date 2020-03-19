import cv2.cv2 as cv
import numpy as np

class SwitchMatcher:
    '''
    SwitchMatch is used to find the best-matched switch ROIs in given proposals.\n
    @
    '''
    def __init__(self, template_path, proposals):
        self.template = self._load_template(template_path)
        self.proposals = proposals

    def _load_template(self, template_path):
        '''
        Load template for a SwitchMatch instance.\n
        @template_path:
            str, template image path
        @return:
            np.array, template image
        '''
        template = cv.imread(template_path, 1)
        template = cv.GaussianBlur(template, (3,3), 5)
        return template
    
    def _get_loss(self, img):
        '''
        Compute loss between given image and template.\n
        @img:
            np.array, input image
        @return:
            float, loss between given image and template
        '''
        img = cv.GaussianBlur(img, (3,3), 5) # smoothing
        img = cv.resize(img, (self.template.shape[1], self.template.shape[0])) # align in size
        
        # vectorization
        img = np.reshape(img, (1,-1))
        template = np.reshape(self.template, (1,-1))

        # normalization
        img = img / 255.0
        img = img - np.mean(img)
        template = template / 255.0
        template = template - np.mean(template)

        loss = np.linalg.norm(img-template)

        return loss

    def _get_variance_smoothness(self, roi):
        '''
        Get variance smoothness of given ROI.\n
        @roi:
            np.array, ROI which you want to compute the pixel variance
        @return:
            float, pixel variance of given ROI
        '''
        rows, cols, _ = roi.shape
        roi = cv.GaussianBlur(roi, (3,3), 1.5)
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        mean = np.array([np.mean(h, dtype=np.float64), np.mean(s, dtype=np.float64), np.mean(v, dtype=np.float64)]).reshape((1,-1))
        losses = []
        for r in range(rows):
            for c in range(cols):
                value = hsv[r, c].reshape((1,-1))
                losses.append(np.linalg.norm(value-mean))
        return np.var(np.array(losses, np.float64))

    def _get_entropy_smoothness(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = cv.calcHist([img], [0], None, [256], (0,256))
        return hist

    def match(self):
        '''
        Match for the most possible switch ROI in proposals.\n
        @return:
            list[dict], list of match results, in loss-ascending order
        '''
        if len(self.proposals) == 0:
            return []
        loss_maps = []
        for i, roi in enumerate(self.proposals):
            loss_maps.append({'loss': self._get_loss(roi), 'id': i})
        sorted_loss_maps = sorted(loss_maps, key=lambda loss_map: loss_map['loss'])
        return sorted_loss_maps
        


if __name__ == "__main__":
    pass