import os
import cv2
import yaml
import json

class CaltechImdb():

    def __init__(self, devkit_path):
        self.path_devkit = devkit_path
        self.img_path = os.path.join(self.path_devkit, 'Images')
        self.annotation_file = os.path.join(self.path_devkit, 'Annotations', 'annotations.json')
        self._annotations = None
        self._img_ext = '.jpg'
        self.img_height = 300
        self.img_width = 300

    @property
    def annotation(self):
        if not self._annotations:
            with open(self.annotation_file, 'rb') as f:
                print 'reading annotations file'
                self._annotations = json.load(f)
            print 'Done reading ann'
        return self._annotations

    def get_annotation(self, image):
        return self.annotation[image]


    def create_lmdb(self, image_set):
        """Create lmdb from given image_set"""

        img_set_path = os.path.join(self.path_devkit, 'ImageSets', image_set + '.txt')

        img_list = []
        with open(img_set_path, 'rb') as f:
            img_list = map(str.strip, f.readlines())

        for im in img_list:
            path = os.path.join(self.img_path, im + self._img_ext)
            annotation_dict = self.get_annotation(im)
            
def test():
    imdb = CaltechImdb('/Users/aambasth/caltech_dataset/target')
    path = imdb.create_lmdb('train')

if __name__ == '__main__':
    test()
