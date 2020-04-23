import os
import glob
import cv2.cv2 as cv
from switch_detection import SwitchDetector
from switch_matcher import SwitchMatcher

HOME_PATH = 'E:/study/final_design'

def rename(path, new_name):
    prefix = path.split('/')
    prefix = prefix[:len(prefix)-1]
    new_prefix = ''
    for p in prefix:
        new_prefix = new_prefix + p + '/'
    new_name = new_prefix+new_name+'.jpg'
    os.rename(path, new_name)
    print('Rename: {} -> {}'.format(path, new_name))

def registrate(paths, start_id):
    for i, path in enumerate(paths):
        img = cv.imread(path, 1)
        img = cv.resize(img, (300, 400))
        cv.imshow('preview', img)
        new_name = ''
        key = cv.waitKey(0)
        if key == ord('r'):
            new_name = str(i+start_id) + '_' + 'r'
        elif key == ord('l'):
            new_name = str(i+start_id) + '_' + 'l'
        else:
            print('error name code!')
            exit()
        rename(path, new_name)
        cv.destroyAllWindows()
    print('Done! Total {} images are registrated!'.format(len(paths)))

def detection_test(paths):
    cols, rows = 768, 1024
    n_correct = 0
    for idx, path in enumerate(paths):
        location = path.split('_')[-1]
        location = location.split('.')[0]

        detector = SwitchDetector(ratio_eps=0.3)

        img = cv.imread(path, 1)
        img = cv.resize(img, (cols, rows))

        # crop for correct area
        if location == 'l':
            img = img[int(rows/3):int(rows/3*2)+1, :int(cols/4)]
        elif location == 'r':
            img = img[int(rows/3):int(rows/3*2)+1, int(cols/4*3):]

        roi, boxes = detector.detect(img)
        real_boxes = detector.check_boxes(boxes)


        proposals = detector.get_proposals(roi, real_boxes)
        matcher = SwitchMatcher(HOME_PATH+'/images/original/2.26_template.jpg', proposals)
        
        loss_maps = matcher.match()
        if len(loss_maps) == 0: # if no ROI detected
            print('No switch ROI detected in {}'.format(path))
            continue
        best_match_id = loss_maps[0]['id']
        print('Min loss: {}'.format(loss_maps[0]['loss']))

        roi = detector.draw_bounding_box(roi, boxes, (255,0,0)) # draw all boxes
        roi_with_box = detector.draw_bounding_box(roi, [real_boxes[best_match_id]]) # draw best matched
        cv.imshow('detection_result', roi_with_box)
        key = cv.waitKey(0)
        if key == ord('t'):
            n_correct += 1
        elif key == ord('f'):
            pass
        elif key == ord('s'):
            cv.imwrite('switch_test_{}.png'.format(idx), roi_with_box)
            print('image saved')
        else:
            print('Wrong key code!')
            quit()
        cv.destroyAllWindows()
    print('----------- Test result -----------')
    print('Total test samples: {}'.format(len(paths)))
    print('Detected: {}'.format(n_correct))
    print('Recall rate: {}'.format(n_correct/len(paths)))

def main():
    paths = os.listdir(HOME_PATH+'/images/switch_test/')
    paths = [HOME_PATH+'/images/switch_test/'+p for p in paths]

    # registrate(paths, 64) # location registration
    detection_test(paths)
    

if __name__ == "__main__":
    main()