from data.data_source_image_folder import DataSourceImageFolder
import numpy as np


class TriDataIter:
    def __init__(self, img_folder, label_npy_file, batch_size=32, worker=8, shuffle=True):
        self.img_folder = img_folder
        self.label_npy_file = label_npy_file
        self.batch_size = batch_size
        self.worker = worker
        self.label_npy = None

    def __iter__(self):
        with open(self.label_npy_file, "rb") as f:
            labels = np.load(f)

        def sort_cb(filenames):
            return filenames.sort(key=lambda f: int(f.split('.')[0]))
        ds = DataSourceImageFolder(self.img_folder, labels, sort=sort_cb)
        return DataIterator(ds, batch_size=self.batch_size, workers=self.workers, shuffle=shuffle):

if __name__ == '__name__':
    img_folder = "/home/woody/data/images"
    label_file = "/home/woody/data/label.npy"
    for img, label in iter(TriDataIter(), label_file):
        print(img.shape)
        print(label.shape)
