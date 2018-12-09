import glob
import math
import os
import shutil

import contextlib2
import tensorflow as tf
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util

"""
The purpose of this script is to create a set of .tfrecords files using
- a folder images/ that contains all images for training and validation in 224x224
- a folder containing corresponding label files for each image

Example of use:
python create_tfrecords.py 
"""


def dict_to_tf_example(image_path, integer_label, box):
    """
    Returns:
        an instance of tf.Example or None.
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    feature = {
        'image': dataset_util.bytes_feature(encoded_image_data),
        'label': dataset_util.int64_feature(integer_label)
    }

    xmin_list, ymin_list, xmax_list, ymax_list = [], [], [], []
    xmin, ymin, xmax, ymax = box

    if (xmin > xmax) or (ymin > ymax) or (xmin > 1.0) or (xmin < 0.0) or (xmax > 1.0) or (xmax < 0.0) \
            or (ymin > 1.0) or (ymin < 0.0)or (ymax > 1.0) or (ymax < 0.0):
        print("Invalid %s " % image_path)
        return None

    xmin_list.append(xmin)
    ymin_list.append(ymin)
    xmax_list.append(xmax)
    ymax_list.append(ymax)

    feature.update({
        'xmin': dataset_util.float_list_feature(xmin_list),
        'ymin': dataset_util.float_list_feature(ymin_list),
        'xmax': dataset_util.float_list_feature(xmax_list),
        'ymax': dataset_util.float_list_feature(ymax_list)
    })

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def read_boxes(label_filename, class_name):
    """
    Returns all boxes from a label file with the given class.
    """
    boxes = []
    try:
        with open(label_filename, "r") as label_file:
            for line in label_file.readlines():
                if line.startswith(class_name):
                    labels = line.strip().split()
                    boxes.append([float(labels[1]), float(labels[2]), float(labels[3]), float(labels[4])])
    except IOError:
        print("Cannot find %s.txt" % label_filename)
    return boxes


def main():
    """
    Write Examples to tfrecord files with a single object/bbox per Example.
    Not super efficient, but does the job and only has to be done once before starting the training.
    """
    NUM_VAL_IMAGES = 500  # number of images for validation
    NUM_SHARDS = 10
    OUTPUT_DIR_TRAIN = "tfrecords/train/"
    OUTPUT_DIR_VAL = "tfrecords/val/"

    shutil.rmtree("tfrecords/", ignore_errors=True)
    os.makedirs(OUTPUT_DIR_TRAIN)
    os.makedirs(OUTPUT_DIR_VAL)

    num_examples_written = 0

    kitti_classes = ["car", "van", "truck", "pedestrian", "person_sitting", "cyclist", "tram", "misc"]

    label_files = glob.glob("%s*.txt" % "labels/")

    # TFRecord set for validation (not sharded as only 500 images)
    val_writer = tf.python_io.TFRecordWriter('%skitti_val.tfrecord' % OUTPUT_DIR_VAL)

    train_dataset_name = '%skitti_train.tfrecord' % OUTPUT_DIR_TRAIN

    with contextlib2.ExitStack() as tf_record_close_stack:
        sharded_train_output = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, train_dataset_name, NUM_SHARDS)

        for kitti_class in kitti_classes:

            for label_file in label_files:
                label_name = os.path.basename(label_file).split(".")[0]
                boxes = read_boxes(label_file, kitti_class)
                image_path = ("images/%s.png " % label_name)
                integer_label = kitti_classes.index(kitti_class)

                for box in boxes:
                    tf_example = dict_to_tf_example(image_path, integer_label, box)

                    if int(label_name) < NUM_VAL_IMAGES:
                        val_writer.write(tf_example.SerializeToString())
                    else:
                        output_shard_index = num_examples_written % NUM_SHARDS
                        sharded_train_output[output_shard_index].write(tf_example.SerializeToString())
                        num_examples_written += 1

    val_writer.close()


if __name__ == "__main__":
    main()
