import os
import argparse
from random import shuffle

def get_annotation_names(annotation_path):
    names = os.listdir(annotation_path)
    names =[name.split('.')[0] for name in names]
    return names

def get_trainval_and_test_names(names, trainval_ratio):
    capacity = int(trainval_ratio * len(names))
    shuffle(names)
    trainvals = names[:capacity]
    tests = names[capacity:]
    return trainvals, tests

def get_train_and_val_names(trainval_names, train_ratio):
    capacity = int(train_ratio * len(trainval_names))
    shuffle(trainval_names)
    trains = trainval_names[:capacity]
    vals = trainval_names[capacity:]
    return trains, vals

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='python generate_voc_labels [annotation_path] [trainval_ratio] [train_ratio] [save_path]',
        description='Generate VOC txt label files.'
        )
    parser.add_argument('annotation_path', help='Folder path of annotation XML files.', type=str)
    parser.add_argument('trainval_ratio', help='Ratio of samples used as train-val data.', type=float)
    parser.add_argument('train_ratio', help='Ratio of samples used as training data from train-val data.', type=float)
    parser.add_argument('save_path', help='Folder path to save txt outputs. They are trainval.txt, train.txt, val.txt and test.txt.', type=str)
    args = parser.parse_args()

    if len(os.sys.argv) != 5:
        print('Wrong argument numbers!')
        parser.print_help()
    else:
        # collect file names
        names = get_annotation_names(args.annotation_path)
        trainval_names, test_names = get_trainval_and_test_names(names, args.trainval_ratio)
        train_names, val_names = get_train_and_val_names(trainval_names, args.train_ratio)

        # generate txt
        with open(os.path.join(args.save_path, 'trainval.txt'), 'w') as f:
            for name in trainval_names:
                f.write('{}\n'.format(name))
        with open(os.path.join(args.save_path, 'test.txt'), 'w') as f:
            for name in test_names:
                f.write('{}\n'.format(name))
        with open(os.path.join(args.save_path, 'train.txt'), 'w') as f:
            for name in train_names:
                f.write('{}\n'.format(name))
        with open(os.path.join(args.save_path, 'val.txt'), 'w') as f:
            for name in val_names:
                f.write('{}\n'.format(name))

        print('Done!')