import os
import cv2.cv2 as cv
import numpy as np

def batch_resize(folder_path, save_path, dst_size=(600,800)):
    img_names = os.listdir(folder_path)
    for i, name in enumerate(img_names):
        img = cv.imread(folder_path+'/'+name, 1)
        if img.shape[0] < img.shape[1]: # for "fat" images
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.resize(img, dst_size)
        cv.imwrite(save_path+'/'+name.split('.')[0]+'.jpg', img)
        print('{}/{} is resized to {}, current {}, total {}'.format(folder_path, name, dst_size, i+1, len(img_names)))
    print('Done!')

if __name__ == "__main__":
    batch_resize('../../images/valves/3.17', '../../images/resized_valves/3.17', (600,800))
