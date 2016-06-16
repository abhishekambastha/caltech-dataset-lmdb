import os
import cv2
import json


class CaltechImdb():

    def __init__(self, devkit_path):
        self.path_devkit = devkit_path
        self.img_path = os.path.join(self.path_devkit, 'Images')
        self.annotation_file = os.path.join(self.path_devkit, 'Annotations', 'annotations.json')
        self._annotations = None
        self._img_ext = '.jpg'
        self.img_height = 300.0
        self.img_width = 300.0


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


    def _parse_annotation_dict(self, annotation_dict):
        bbox_list = annotation_dict['coords_list']
        return bbox_list


    def _normalize_bboxes(self, bboxes):
        normalized_bboxes = []
        for box in bboxes:
            normalized_box = {}
            normalized_box['x1'] = (1.0 * box['x1']) / (self.img_width)
            normalized_box['x2'] = (1.0 * box['x2']) / (self.img_width)
            normalized_box['y1'] = (1.0 * box['y1']) / (self.img_height)
            normalized_box['y2'] = (1.0 * box['y2']) / (self.img_height)
            normalized_bboxes.append(normalized_box)

        return normalized_bboxes


    def create_lmdb(self, image_set):
        """Create lmdb from given image_set"""

        img_set_path = os.path.join(self.path_devkit, 'ImageSets', image_set + '.txt')

        img_list = []
        with open(img_set_path, 'rb') as f:
            img_list = map(str.strip, f.readlines())

        self.img_height, self.img_width, _ = (cv2.imread(os.path.join(self.img_path, img_list[0] + self._img_ext))).shape

        for im in img_list:
            path = os.path.join(self.img_path, im + self._img_ext)
            annotation_dict = self.get_annotation(im)
            bboxes = self._parse_annotation_dict(annotation_dict)
            normal_bboxes = self._normalize_bboxes(bboxes)
            for box in normal_bboxes:
                print 'x1: {}, y1: {}, x2: {}, y2: {}'.format(box['x1'], box['y1'], box['x2'], box['y2'])

##
##TODO: import proto file and populate lmdb file
## Come back and check the normalization method


def test():
    imdb = CaltechImdb('/Users/aambasth/caltech_dataset/target')
    path = imdb.create_lmdb('train')


if __name__ == '__main__':
    test()
