import os
import lmdb
import numpy as np
import matplotlib.pyplot as plt
import py.datum_pb2 as datum_pb2
import sys
from StringIO import StringIO
import PIL.Image
import cv2

caffe_root = '/home3/aambasth/abhishek/ssd/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


LMDB_PATH='/home3/aambasth/abhishek/ssd/caffe/examples/VOC0712/VOC0712_test_lmdb'

env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
visualize = False

datum = datum_pb2.AnnotatedDatum()
cv2.startWindowThread()
cv2.namedWindow('preview')
with env.begin() as txn:
    cur = txn.cursor()
    for i in xrange(1):
        if not cur.next():
            cur.first()

        key, value = cur.item()
        datum.ParseFromString(value)
        s = StringIO()
        s.write(datum.datum.data)
        s.seek(0)
        img_alt = np.asarray(PIL.Image.open(s))
        print 'datum start'
        print datum
        print 'datum end'
        for anno in datum.annotation_group:
            print anno

        if visualize:
            cv2.imshow('preview', img_alt)
            cv2.waitKey(0)

    if visualize:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
