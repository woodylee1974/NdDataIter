import numpy as np
import os
import glob
from .data_source import DataSource
import cv2


class DataSourceImageFolder(DataSource):
    def __init__(self, img_path, img_label=None, preprocess=None, sort=None):
        self.preprocess = preprocess
        img_names = glob.glob(os.path.join(img_path, "*.jpg"))
        img_names += glob.glob(os.path.join(img_path, "*.JPG"))
        img_names += glob.glob(os.path.join(img_path, "*.png"))
        img_names += glob.glob(os.path.join(img_path, "*.PNG"))
        if sort is None:
            pass
        elif callable(sort):
            img_names = sort(img_names)
        elif sort:
            img_names.sort()
        self.img_label = None
        if isinstance(img_label, str) and os.path.exists(img_label):
            with open(img_label, "r") as input_file:
                lines = input_file.readlines()
                self.img_label = [(int(line)-1) for line in lines]
        elif isinstance(img_label, list):
            self.img_label = np.array(self.img_label)
        elif isinstance(img_label, np.ndarray):
            self.img_label = img_label
        elif img_label is None:
            pass
        else:
            raise ValueError("img_label is invalid, it should be a path, list,  N-D array or None")

        self.img_list = img_names

    def __len__(self):
        if self.img_label is None:
            return len(self.img_list)
        return min(len(self.img_list), len(self.img_label))

    def __getitem__(self, item):
        img = cv2.imread(self.img_list[item])
        if callable(self.preprocess):
            img = self.preprocess(img)
        if self.img_label is not None:
            label = self.img_label[item]
            return img, label
        else:
            return img


if __name__ == '__main__':
    ds = DataSourceImageFolder('/home/woody/work/deep/models/research/VOCdevkit/VOC2012/JPEGImages')
    print(len(ds))
    for i in range(10):
        img = ds[i]
        print(img.shape)