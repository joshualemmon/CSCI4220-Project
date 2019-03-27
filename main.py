import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import argparse
import sqlite3
from collections import Counter

ROOT_DIR = os.path.abspath(".")
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))  # To find local version
import coco

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1 

def load_model():
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")
	config = InferenceConfig()
	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
		utils.download_trained_weights(COCO_MODEL_PATH)
	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	return model, COCO_MODEL_PATH

def main(args):
	model, COCO_MODEL_PATH = load_model()
	print("created model")
	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)
	print("loaded model")
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

	db = sqlite3.connect(args.db_path)
	if(args.proc_imgs):
		img_labels = analyze_images(model,args.img_dir, args.num_imgs, class_names, args.output)
		for i, c in img_labels:
			rel = calc_img_relevence(c)
			add_labels_db(db, i, rel)

	if(args.proc_vids):
		pass
		

# Analyzes images in image directory and detcts objects in each
def analyze_images(model, img_dir, num_imgs, class_names, output):
	img_names = []
	images = []
	for i, f in enumerate(os.listdir(img_dir), 1):
		if i <= num_imgs or num_imgs == -1:
			img_names.append(f)
		else:
			break
	results = []
	for i in img_names:
		image = skimage.io.imread(img_dir + '/' + i)	
		result = model.detect([image], verbose=1)
		r = result[0]
		results.append(r)
		if output:
			visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
	class_ids = []
	for r in results:
		class_ids.append(r['class_ids'])
	img_labels = zip(img_names, class_ids)
	return img_labels

def analyze_videos(model, vid_dir, num_vids, class_names, output):
	vid_names = []
	videos = []
	for i, f in enumerate(os.listdir(args.vid_dir)):
		if i <= num_vids or num_vids == -1:
			vid_names.append(f)
		else:
			break
	for f in file_names:
		videos.append(skimage.io.vread(args.vid_dir + '/' + f))
	print(len(videos))

# Returns relevence for each class id as a dict
def calc_img_relevence(class_ids):
	rel = dict()
	id_set = set(class_ids)
	count = Counter(class_ids)
	for i in id_set:
		rel[i] = count[i]/len(class_ids)
	return rel

# Returns relevence for each class id as a dict
def calc_vid_relevence(class_ids):
	rel = dict()
	pass

# Writes class id and relevence to db for given file
def add_labels_db(db, fname, rel):
	c = db.cursor()
	c.execute("select file_id from files where filename = '" + fname + "'")
	file_id = c.fetchone()[0]
	c.execute("select count(file_id) from file_label where file_id = " + str(file_id))
	if(c.fetchone()[0] == 0):
		for key in rel.keys():
			c.execute("insert or ignore into file_label(file_id, label_id, relevance) values(" + str(file_id) + ", " + str(key) + ", " + str(rel[key]) + ")")
	db.commit()


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-id', '--img_dir', type=str, default='./images')
	argparser.add_argument('-pi', '--proc_imgs', type=bool, default=True)
	argparser.add_argument('-vd', '--vid_dir', type=str, default='./videos')
	argparser.add_argument('-pv', '--proc_vids', type=bool, default=False)
	argparser.add_argument('-o', '--output', type=bool, default=False)
	argparser.add_argument('-ni', '--num_imgs', type=int, default=-1)
	argparser.add_argument('-nv', '--num_vids', type=int, default=-1)
	argparser.add_argument('-db', '--db_path', type=str, default='labeldb.db')
	main(argparser.parse_args())