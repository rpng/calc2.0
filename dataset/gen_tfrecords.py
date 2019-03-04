#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
import tensorflow as tf
from time import time


vw = 320
vh = 320

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
   
    tf.app.flags.DEFINE_string("output_dir", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/calc_tfrecords/", "")
    tf.app.flags.DEFINE_string("coco_root", "/mnt/f3be6b3c-80bb-492a-98bf-4d0d674a51d6/coco/", "")
    tf.app.flags.DEFINE_integer("num_files", 100, "Num files to write for train dataset. More files=better randomness")
    tf.app.flags.DEFINE_boolean("debug", False, "")
    

    if FLAGS.debug:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch
        plt.ion()
        imdata = None

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate():
    try:
        from . import coco
        from .coco_classes import coco_classes, calc_classes, calc_class_names
    except:
        import coco
        from coco_classes import coco_classes, calc_classes, calc_class_names
    import cv2
    if not os.path.isdir(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    train_writers = []
    for ii in range(FLAGS.num_files):
        train_writers.append(None if FLAGS.debug else \
                tf.python_io.TFRecordWriter(FLAGS.output_dir + "train_data%d.tfrecord" % ii))
    
    val_writer = None if FLAGS.debug else \
            tf.python_io.TFRecordWriter(FLAGS.output_dir + "validation_data.tfrecord")


    nclasses = len(calc_classes.keys())

    class_percents = np.zeros((nclasses), dtype=np.float32)
    for split, writer in [('train', train_writers), ('val', val_writer)]:
        # Load dataset
        dataset = coco.CocoDataset()
        dataset.load_coco(FLAGS.coco_root, split)

        # Must call before using the dataset
        dataset.prepare()

        print("Image Count: {}".format(len(dataset.image_ids)))
        print("COCO Class Count: {}".format(dataset.num_classes))
        print("CALC Class Count: {}".format(nclasses))
        #for i, info in enumerate(dataset.class_info):
        #    print("{:3}. {:50}".format(i, info['name']))
        
        count = 1
        for image_id in dataset.image_ids:
            print("Working on sample %d" % image_id)
            if split=='val':
                cl_live = cv2.cvtColor(cv2.resize(
                    cv2.imread("CampusLoopDataset/live/Image%s.jpg" % (str(count).zfill(3))),
                    (vw, vh), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB)
                cl_mem = cv2.cvtColor(cv2.resize(
                    cv2.imread("CampusLoopDataset/memory/Image%s.jpg" % (str(count).zfill(3))),
                    (vw, vh), interpolation=cv2.INTER_CUBIC),
                    cv2.COLOR_BGR2RGB)

            image = cv2.resize(dataset.load_image(image_id),
                (vw, vh), interpolation=cv2.INTER_CUBIC)
            masks, class_ids = dataset.load_mask(image_id)
            mask_label = np.zeros((vh, vw, nclasses), dtype=np.bool)
            for i in range(masks.shape[2]):
                cid = calc_classes[coco_classes[class_ids[i]][1]]
                    
                mask_label[:, :, cid] = np.logical_or(mask_label[:,:,cid], 
                        cv2.resize(masks[:, :, i].astype(np.uint8), (vw, vh), 
                        interpolation=cv2.INTER_NEAREST).astype(np.bool))

            # No labels for BG. Make them!
            mask_label[:, :, 0] = np.logical_not(np.any(mask_label[:, :, 1:], axis=2))    
            if split=='train':
                cp = np.mean(mask_label, axis=(0,1))
                class_percents += (1.0 / count) * (cp - class_percents)
            mask = np.argmax(mask_label, axis=-1)

            if FLAGS.debug:
                rgb = np.zeros((vh, vw, 3))

                legend = []
                np.random.seed(0)
                for i in range(nclasses):
                    c = np.random.rand(3)
                    case = mask==i
                    if np.any(case):
                        legend.append(Patch(facecolor=tuple(c), edgecolor=tuple(c),
                                    label=calc_class_names[i]))

                    rgb[case, :] = c
                
                _image = cv2.resize(image, (vw, vh)) / 255.0

                _image = 0.3 * _image + 0.7 * rgb

                global imdata
                if imdata is None:
                    imdata = plt.imshow(_image)
                    f = plt.gca()
                    f.axes.get_xaxis().set_ticks([])
                    f.axes.get_yaxis().set_ticks([])
                else:
                    imdata.set_data(_image)

                lgd = plt.legend(handles=legend, loc='upper left', bbox_to_anchor=(1.0, 1))
                
                plt.pause(1e-9)
                plt.draw()
                plt.pause(3)

            else:
                features_ = {
                    'img': bytes_feature(tf.compat.as_bytes(image.tostring())),
                    'label': bytes_feature(tf.compat.as_bytes(mask.astype(np.uint8).tostring()))
                }

                if split=='val':
                    features_['cl_live'] = bytes_feature(tf.compat.as_bytes(cl_live.tostring())),
                    features_['cl_mem'] = bytes_feature(tf.compat.as_bytes(cl_mem.tostring())),

                example = tf.train.Example(features=tf.train.Features(feature=features_))
                        

                if split=='val':
                    writer.write(example.SerializeToString())
                else:
                    writer[np.random.randint(0,FLAGS.num_files)].write(example.SerializeToString())
            
            if split=='val' and image_id==99:
                break
            count += 1
    class_weights = 1.0 / class_percents 
    with open('loss_weights.txt', 'w') as f:
        s = ''
        for w in class_weights:
            s += str(w) + ' '
        f.write(s)

def main(argv):
    del argv
    generate()


if __name__ == "__main__":
    tf.app.run()
