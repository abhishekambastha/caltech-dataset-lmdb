import os
import random
import lmdb
import cv2
import json
import py.datum_pb2 as datum_pb2
import numpy as np


class CaltechImdb():

    def __init__(self, devkit_path):
        self.path_devkit = devkit_path
        self.img_path = os.path.join(self.path_devkit, 'Images')
        self.annotation_file = os.path.join(self.path_devkit, 'Annotations', 'annotations.json')
        self._annotations = None
        self._img_ext = '.jpg'
        self.img_height = None
        self.img_width = None

    @property
    def annotation(self):
        if not self._annotations:
            with open(self.annotation_file, 'rb') as f:
                print 'reading annotations file'
                self._annotations = json.load(f)
            print 'Loaded annotations from {}'.format(self.annotation_file)
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

    def _prepare_dataset(self, image_set):

        img_set_path = os.path.join(self.path_devkit, 'ImageSets', image_set + '.txt')
        img_list = []
        with open(img_set_path, 'rb') as f:
            img_list = map(str.strip, f.readlines())

        self.img_height, self.img_width, _ = (cv2.imread(os.path.join(self.img_path, img_list[0] + self._img_ext))).shape

        return img_list

    def _get_image_path(self, img):
        return os.path.join(self.path_devkit, 'Images', img + self._img_ext)

    def test_image_anno(self, im):
        img_path = os.path.join(self.path_devkit, 'Images', im + self._img_ext)
        img = cv2.imread(self._get_image_path(im))
        anno = self.get_annotation(im)
        bboxes = self._parse_annotation_dict(anno)
        for box in bboxes:
            x1 = box['x1']
            y1 = box['y1']
            x2 = box['x2']
            y2 = box['y2']
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 1)
        print 'Showing the image {}'.format(im)
        return img


    def create_lmdb(self, image_set):
        """Create lmdb from given image_set"""

        img_list = self._prepare_dataset(image_set)

        # map_size = self.img_width * self.img_height * 8 * len(img_list)
        map_size = 1099511627776
        print 'Initialized lmdb with size {}'.format(map_size)

        env = lmdb.open(image_set + '.lmdb', map_size=map_size)

        num_images = len(img_list) * 1.0
        mean_image = np.zeros_like(cv2.imread(self._get_image_path(img_list[0])), dtype=np.float32)

        with env.begin(write=True) as txn:
            for i, im in enumerate(img_list):

                annotated_datum = datum_pb2.AnnotatedDatum()
                img_file = cv2.imread(self._get_image_path(im))
                mean_image += (1000.0 / num_images) * img_file
                annotated_datum.datum.data = cv2.imencode('.jpg', img_file)[1].tostring()

                annotated_datum.datum.label = -1
                annotated_datum.datum.encoded = True
                annotated_datum.type = datum_pb2.AnnotatedDatum.BBOX

                annotation_dict = self.get_annotation(im)
                bboxes = self._parse_annotation_dict(annotation_dict)
                normal_bboxes = self._normalize_bboxes(bboxes)

                #remove invalid images
                bbox_present = len(bboxes) > 0
                valid_image = True

                for box in normal_bboxes:
                    annotation_group = annotated_datum.annotation_group.add()

                    #using pascal dataset's label
                    annotation_group.group_label = 1
                    annotation = annotation_group.annotation.add()
                    annotation.instance_id = 0
                    datum_bbox = annotation.bbox

                    datum_bbox.xmin = box['x1']
                    datum_bbox.ymin= box['y1']
                    datum_bbox.xmax = box['x2']
                    datum_bbox.ymax = box['y2']
                    datum_bbox.difficult = False

                    #remove invalid images
                    if bbox_present:
                        valid_image &= box['x1'] >= 0 and box['x1'] <=1
                        valid_image &= box['x2'] >= 0 and box['x2'] <=1
                        valid_image &= box['y1'] >= 0 and box['y1'] <=1
                        valid_image &= box['y2'] >= 0 and box['y2'] <=1

                if valid_image:
                    str_id = '{:08}'.format(i)
                    txn.put(str_id.encode('ascii'), annotated_datum.SerializeToString())
                    print '[INFO]: Added image {} with {} bboxes'.format(im, len(bboxes))
                else:
                    print '[INFO]: Filtered image {}'.format(im)

        cv2.imwrite(image_set + '_mean.jpg' ,mean_image/1000)



##
##TODO: import proto file and populate lmdb file
## Come back and check the normalization method

def sub_sample(image_set):
    devkit_path = '/home3/aambasth/abhishek/dataset-convertor/target/ImageSets'
    file_path = os.path.join(devkit_path, image_set + '.txt')
    img_list = []
    final_list = []
    with open(file_path, 'rb') as f:
        img_list = f.readlines()

    num_images = len(img_list)
    for i in xrange(num_images / 30):
        temp_list = img_list[30*i: 30*(i+1)]
        im = random.choice(temp_list)
        final_list.append(im)

    print 'Final List size ', len(final_list)
    with open('filter.txt', 'w') as f:
        for img in final_list:
            f.write('%s' %img)




def test():
    devkit_path = '/home3/aambasth/abhishek/dataset-convertor/target'
    imdb = CaltechImdb(devkit_path)
    path = imdb.create_lmdb('sub_train')
    # TODO: test data


def testing():
    img_name = raw_input()
    devkit_path = '/home3/aambasth/abhishek/dataset-convertor/target'
    imdb = CaltechImdb(devkit_path)
    res = imdb.test_image_anno(img_name)
    cv2.startWindowThread()
    cv2.namedWindow('im')
    cv2.imshow('im', res)
    cv2.waitKey(0)



if __name__ == '__main__':
    test()
    # testing()
    # sub_sample('test')
