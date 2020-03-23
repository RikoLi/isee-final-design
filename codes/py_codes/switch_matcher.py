import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # for test

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
        return template
    
    def _get_loss_l2(self, img):
        '''
        Compute L2-loss between given image and template.\n
        @img:
            np.array, input image
        @return:
            float, L2-loss between given image and template
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

    def _get_entropy_single(self, roi):
        hist = cv.calcHist([roi], [0], None, [256], (0,256))
        hist = [h[0] for h in hist.tolist()] # distribution
        prob = [h / sum(hist) for h in hist] # probability
        entropy = sum([-p * np.log2(p) for p in prob if p != 0]) # entropy
        return entropy

    def _get_mean_single(self, roi):
        return np.mean(roi)

    def _get_variance_single(self, roi):
        return np.var(roi)

    def _get_smoothness_single(self, roi):
        return 1 / (1 + self._get_variance_single(roi))

    def _get_ncc_similarity_single(self, roi):
        # smoothing
        template = cv.GaussianBlur(self.template, (3,3), 1)

        # align in shape
        roi = cv.resize(roi, (template.shape[1], template.shape[0]))

        # mean reduce
        template = template - np.mean(template)
        roi = roi - np.mean(roi)

        # compute relation factor
        numerator = np.sum(np.multiply(template, roi))
        denominator = np.sqrt(np.sum(np.square(template))) * np.sqrt(np.sum(np.square(roi)))

        # return numerator / denominator
        return denominator / numerator # loss version

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
            # loss_maps.append({'loss': self._get_loss_l2(roi), 'id': i})
            loss_maps.append({'loss': self._get_ncc_similarity_single(roi), 'id': i})
        sorted_loss_maps = sorted(loss_maps, key=lambda loss_map: loss_map['loss'])
        return sorted_loss_maps
        


if __name__ == "__main__":
    img = cv.imread('../../images/original/2.26_template.jpg', 0)
    img = cv.GaussianBlur(img, (3,3), 1.0)
    matcher = SwitchMatcher('test', [])
    out = matcher._get_entropy_single(img)
    print(out)
    # plt.hist(hist, 256, (0,256))
    # plt.show()