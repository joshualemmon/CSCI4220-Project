import os
import sys
import random
import math
import numpy as np
import cv2
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
import argparse
import sqlite3
from collections import Counter
import colorsys	

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

	# Fill in image and video relevance data to the database
	db = sqlite3.connect(args.db_path)
	output_array = [args.bound_boxes, args.masks, args.scores, args.labels]
	if(args.proc_imgs):
		img_labels = analyze_images(model,args.img_dir, args.num_imgs, class_names, args.output, args.save, output_array)
		for i, c, s in img_labels:
			rel = calc_relevence(c, s) 
			add_labels_db(db, i, rel)

	if(args.proc_vids):
		vid_labels = analyze_videos(model,args.vid_dir, args.num_vids, class_names, args.output, args.save, output_array)
		for i, c, s in vid_labels:
			rel = calc_relevence(c, s)
			add_labels_db(db, i, rel) 
		

# Analyzes images in image directory and detects objects in each
def analyze_images(model, img_dir, num_imgs, class_names, output, save, o_arr):
	img_names = []
	for i, f in enumerate(os.listdir(img_dir), 1):
		if i <= num_imgs or num_imgs == -1:
			img_names.append(f)
		else:
			break
	
	# Obtain class_ids from images + visualize
	class_ids = []
	scores = []
	for i in img_names:
		image = skimage.io.imread(img_dir + '/' + i)	
		result = model.detect([image])
		r_class_ids = result[0]['class_ids']
		r_scores = result[0]['scores']
		class_ids.append(r_class_ids)
		scores.append(r_scores)
		if save or output:
			p_img = get_processed_image(image, r_class_ids, boxes=result[0]['rois'], masks=result[0]['masks'], class_names=class_names, scores=r_scores, o_arr=o_arr)
			if output:
				skimage.io.imshow(p_img)
				plt.show()
			if save:
				skimage.io.imshow(p_img)
				plt.savefig('./processed/processed_' + i)
				#skimage.io.imsave('./processed/processed_' + i, p_img)
	
	img_labels = zip(img_names, class_ids, scores)
	return img_labels

# Analyzes videos in video directory and detects objects in each
# Todo: Next step, save final frames with visualize to create short frame video
# Todo: Optimize step, skip similar frames
def analyze_videos(model, vid_dir, num_vids, class_names, output, save, o_arr):
	vid_names = []
	for i, f in enumerate(os.listdir(vid_dir)):
		if i < num_vids or num_vids == -1:
			vid_names.append(f)
		else:
			break

	# Select every nth frame and obtain class_ids for each video
	class_ids = []
	scores = []
	vid_class_ids = []
	vid_scores = []
	
	for v in vid_names:
		cap = cv2.VideoCapture(vid_dir + '/' + v)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter('./processed/processed_' + v, fourcc, 2, (1920, 1080))
		total_frames = int(cap.get(7))
		frame_offset = 30 # Could take in as arg.parse		

		for i in range(0, total_frames, frame_offset):
			cap.set(1, i)
			ret, frame = cap.read()
			result = model.detect([frame])

			r_class_ids = result[0]['class_ids']
			r_scores= result[0]['scores']
			vid_class_ids.extend(r_class_ids)
			vid_scores.extend(r_scores)

			if save:
				p_img = get_processed_image(frame, r_class_ids, boxes=result[0]['rois'], masks=result[0]['masks'], class_names=class_names, scores=r_scores, o_arr=o_arr)
				out.write(p_img)
		
		class_ids.append(vid_class_ids)
		vid_class_ids = []
		scores.append(vid_scores)
		vid_scores = []
		
		cap.release()
		out.release()
		
	# print(class_ids)
	# print(scores)
	vid_labels = zip(vid_names, class_ids, scores)
	return vid_labels

# Returns relevence for each class id as a dict
def calc_relevence(class_ids, scores):
	rel = dict()
	id_set = set(class_ids)
	avg_scores, count = calc_mean_score(class_ids, scores)
	for i in id_set:
		rel[i] = count[i]*avg_scores[i]/len(class_ids)
	return rel

def calc_mean_score(class_ids, scores):
	avg_scores = dict()
	count = Counter(class_ids)
	for i in range(len(class_ids)):
		if class_ids[i] in avg_scores.keys():
			avg_scores[class_ids[i]] += scores[i]
		else:
			avg_scores[class_ids[i]] = scores[i]
	for key in avg_scores:
		avg_scores[key] /= count[key]
	return avg_scores, count


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

# Returns an image object with optional bounding boxes, masks and class names
def get_processed_image(image, class_ids,o_arr, boxes, masks, class_names,
                      scores, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    # Number of instances
    N = class_ids.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if  o_arr[0]:
	        if not np.any(boxes[i]):
	            # Skip this instance. Has no bbox. Likely lost in image cropping.
	            continue
	        y1, x1, y2, x2 = boxes[i]
	        if show_bbox:
	            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
	                                alpha=0.7, linestyle="dashed",
	                                edgecolor=color, facecolor='none')
	            ax.add_patch(p)

        # Label
        if o_arr[3]:
	        if not captions:
	            class_id = class_ids[i]
	            if o_arr[2]:
	            	score = scores[i] if scores is not None else None
	            else:
	            	score = None
	            label = class_names[class_id]
	            caption = "{} {:.3f}".format(label, score) if score else label
	        else:
	            caption = captions[i]
	        ax.text(x1, y1 + 8, caption,
	                color='w', size=11, backgroundcolor="none")

        # Mask
        if o_arr[1]:
	        mask = masks[:, :, i]
	        if show_mask:
	            masked_image = visualize.apply_mask(masked_image, mask, color)

	        # Mask Polygon
	        # Pad to ensure proper polygons for masks that touch image edges.
	        padded_mask = np.zeros(
	            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
	        padded_mask[1:-1, 1:-1] = mask
	        contours = visualize.find_contours(padded_mask, 0.5)
	        for verts in contours:
	            # Subtract the padding and flip (y, x) to (x, y)
	            verts = np.fliplr(verts) - 1
	            p = Polygon(verts, facecolor="none", edgecolor=color)
	            ax.add_patch(p)
    return masked_image.astype(np.uint8)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-id', '--img_dir', type=str, default='./images')
	argparser.add_argument('-pi', '--proc_imgs', action='store_true')
	argparser.add_argument('-vd', '--vid_dir', type=str, default='./videos')
	argparser.add_argument('-pv', '--proc_vids', action='store_true')
	argparser.add_argument('-o', '--output', action='store_true')
	argparser.add_argument('-ni', '--num_imgs', type=int, default=-1)
	argparser.add_argument('-nv', '--num_vids', type=int, default=-1)
	argparser.add_argument('-db', '--db_path', type=str, default='labeldb.db')
	argparser.add_argument('-s', '--save', action='store_true')
	argparser.add_argument('-bb', '--bound_boxes', action='store_true')
	argparser.add_argument('-m', '--masks', action='store_true')
	argparser.add_argument('-sc', '--scores', action='store_true')
	argparser.add_argument('-l', '--labels', action='store_true')

	main(argparser.parse_args())