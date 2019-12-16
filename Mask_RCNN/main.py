from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
import numpy as np
from numpy import zeros
from numpy import asarray
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from natsort import natsorted, ns

import argparse

# %matplotlib inline

from os import listdir
from xml.etree import ElementTree

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--command', required=False, default='predict', help='set train or test for what you want to do')
args = vars(parser.parse_args())

class HandDataset(Dataset):
  # load dataset definitions
  def load_dataset(self, dataset_dir, is_train=True):
    # add classes. we have only one class to add
    self.add_class('dataset', 1, 'hand')
    
    # define data locations for images and annotations
    images_dir = dataset_dir + '\\images\\'
    annotations_dir = dataset_dir + '\\annots\\'

    # iterate through all files in the folder to add class, images, and annotations
    for filename in listdir(images_dir):
      # extract image id
      image_id = filename[:-4]

      # skip bad images
      bad_images = ['00090']
      if image_id in bad_images:
          continue

      TRAINING_SIZE = 4000
      
      # skip all images a fter 150 if we are building the training set
      if is_train and int(image_id) >= TRAINING_SIZE:
        continue
      
      # skip all images before 150 if we are building the test/val set
      if not is_train and int(image_id) < TRAINING_SIZE:
        continue
    
      # setting image file
      img_path = images_dir + filename

      # setting annotation file
      ann_path = annotations_dir + image_id + '.xml'

      # adding images and annotations to dataset
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

  # extract bounding boxes from an annotation file
  def extract_boxes(self, filename):
    # load and parse the file
    tree = ElementTree.parse(filename)

    # get root of document
    root = tree.getroot()

    # extract each bounding box
    boxes = []

    for box in root.findall('.//bndbox'):
      xmin = int(box.find('xmin').text)
      ymin = int(box.find('ymin').text)
      xmax = int(box.find('xmax').text)
      ymax = int(box.find('ymax').text)
      coors = [xmin, ymin, xmax, ymax]
      boxes.append(coors)

    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

  # load the masks for an image
  """Generate instance masks for an image.
    Returns:
      masks: A bool array of shape [height, width, instance count] with one mask per instance
      class_ids: a 1D array of class IDs of the instance masks
  """
  def load_mask(self, image_id):
    # get details of image
    info = self.image_info[image_id]

    # define annotation file location
    path = info['annotation']

    # load XML
    boxes, w, h = self.extract_boxes(path)

    # create one array for all masks, each on a different channel
    masks = zeros([h,  w, len(boxes)], dtype='uint8')

    # create masks
    class_ids = []
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = box[1], box[3]
      col_s, col_e = box[0], box[2]
      masks[row_s:row_e, col_s:col_e, i] = 1 # rectangular masks
      class_ids.append(self.class_names.index('hand'))
    
    return masks, asarray(class_ids, dtype='int32')

  # load an image reference
  def image_reference(self, image_id):
    info = self.image_info[image_id]
    print(info)
    return info['path']

class KangarooDataset(Dataset):
  # load dataset definitions
  def load_dataset(self, dataset_dir, is_train=True):
    # add classes. we have only one class to add
    self.add_class('dataset', 1, 'kangaroo')
    
    # define data locations for images and annotations
    images_dir = dataset_dir + '\\images\\'
    annotations_dir = dataset_dir + '\\annots\\'

    # iterate through all files in the folder to add class, images, and annotations
    for filename in listdir(images_dir):
      # extract image id
      image_id = filename[:-4]

      # skip bad images
      if image_id in ['00090']:
          continue

      # skip all images a fter 150 if we are building the training set
      if is_train and int(image_id) >= 150:
        continue
      
      # skip all images before 150 if we are building the test/val set
      if not is_train and int(image_id) < 150:
        continue
    
      # setting image file
      img_path = images_dir + filename

      # setting annotation file
      ann_path = annotations_dir + image_id + '.xml'

      # adding images and annotations to dataset
      self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

  # extract bounding boxes from an annotation file
  def extract_boxes(self, filename):
    # load and parse the file
    tree = ElementTree.parse(filename)

    # get root of document
    root = tree.getroot()

    # extract each bounding box
    boxes = []

    for box in root.findall('.//bndbox'):
      xmin = int(box.find('xmin').text)
      ymin = int(box.find('ymin').text)
      xmax = int(box.find('xmax').text)
      ymax = int(box.find('ymax').text)
      coors = [xmin, ymin, xmax, ymax]
      boxes.append(coors)

    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height

  # load the masks for an image
  """Generate instance masks for an image.
    Returns:
      masks: A bool array of shape [height, width, instance count] with one mask per instance
      class_ids: a 1D array of class IDs of the instance masks
  """
  def load_mask(self, image_id):
    # get details of image
    info = self.image_info[image_id]

    # define annotation file location
    path = info['annotation']

    # load XML
    boxes, w, h = self.extract_boxes(path)

    # create one array for all masks, each on a different channel
    masks = zeros([h,  w, len(boxes)], dtype='uint8')

    kango_id = self.class_names.index('kangaroo')

    # create masks
    class_ids = []
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = box[1], box[3]
      col_s, col_e = box[0], box[2]
      masks[row_s:row_e, col_s:col_e, i] = 1 # rectangular masks
      class_ids.append(self.class_names.index('kangaroo'))
    
    return masks, asarray(class_ids, dtype='int32')

  # load an image reference
  def image_reference(self, image_id):
    info = self.image_info[image_id]
    print(info)
    return info['path']

class myMaskRCNNConfig(Config):
  #give the configuration a recognizable name
  NAME = "MaskRCNN_config"
  
  # set the number of GPUs to use along with number of images per GPU  
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

  # number of classes (we would normally add +1 for the background): kangaroo + BG
  NUM_CLASSES = 1+1

  # number of training steps per epoch
  STEPS_PER_EPOCH = 100

  # learning rate
  LEARNING_RATE = 0.0002

  # skip detections with < 90% confidence
  DETECTION_MIN_CONFIDENCE = 0.9

  # setting max ground truth instances
  MAX_GT_INSTANCES = 10

if __name__ == '__main__':  
  command = args['command']

  config = myMaskRCNNConfig()
  config.display()

  """
  # prepare train set
  train_set = KangarooDataset()
  train_set.load_dataset('..\\Kangaroo\\kangaroo-master', is_train=True)
  train_set.prepare()
  print('Train: %d' % len(train_set.image_ids))

  # prepare test/val set
  test_set = KangarooDataset()
  test_set.load_dataset('..\\Kangaroo\\kangaroo-master')
  test_set.prepare()
  print('Test: %d' % len(test_set.image_ids))
  """

  # prepare train set
  train_set = HandDataset()
  train_set.load_dataset('E://egohands_kitti_formatted', is_train=True)
  train_set.prepare()
  print('Train: %d' % len(train_set.image_ids))

  # prepare test/val set
  test_set = HandDataset()
  test_set.load_dataset('E://egohands_kitti_formatted')
  test_set.prepare()
  print('Test: %d' % len(test_set.image_ids))


  print('Loading Mask R-CNN model...')
  files = [f for f in os.listdir('E://mask_rcnn/weights')]
  files = natsorted(files, key=lambda y: y.lower())
  new_ind = len(files)
  model_path = 'E://mask_rcnn/weights/' + files[-1]
  print('Model path: {}'.format(model_path))
  if command == 'train':
    """
    model = modellib.MaskRCNN(mode='training', config=config, model_dir='./')

    # load weights for COCO
    model.load_weights('.\\mask_rcnn_coco.h5', by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # train heads with higher learning rate to speed up learning
    model.train(train_set, test_set, learning_rate=2 * config.LEARNING_RATE, epochs=5, layers='heads')

    history = model.keras_model.history.history

    # save trained weights
    model_path = '..\\Kangaroo\\kangaroo-master\\mask_rcnn_' + '.' + str(time.time()) + '.h5'
    
    model.keras_model.save_weights(model_path)
    """
  
    model = modellib.MaskRCNN(mode='training', config=config, model_dir='E://mask_rcnn')

    # load weights for COCO
    model.load_weights(model_path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # train heads with higher learning rate to speed up learning
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')

    history = model.keras_model.history.history

    # save trained weights
    model_path = 'E://mask_rcnn/weights/mask_rcnn_v' + str(new_ind) + '.h5'

    model.keras_model.save_weights(model_path)
  
  elif command == 'predict':
    # loading model in the inference mode
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir='./')

    # loading the trained weights of the custom dataset
    model.load_weights(model_path, by_name=True)

    direc = 'E://egohands_kitti_formatted/extras/'
    image_paths = [f for f in os.listdir(direc)]

    for path in image_paths:
      image = cv2.imread(os.path.join(direc, path))
      results = model.detect([image], verbose=1)
      import pdb; pdb.set_trace()
      res = results[0]
      visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], test_set.class_names, res['scores'], title='predictions')

  else:
    # detect objects in image with masks and bounding box from trained model

    # loading model in the inference mode
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir='./')

    # loading the trained weights of the custom dataset
    model.load_weights(model_path, by_name=True)

    # display
    for image_id in range (0, 200):
      image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, config, image_id, use_mini_mask=False)

      info = test_set.image_info[image_id]

      print('image ID: {}.{}  ({}) {}'.format(info['source'], info['id'],  image_id, test_set.image_reference(image_id)))

      results = model.detect([image], verbose=1)
      # display results
      res = results[0]

      if res['rois'].size != 0:
        import pdb; pdb.set_trace()
        visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], test_set.class_names, res['scores'], title='predictions')
    