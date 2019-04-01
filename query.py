import sqlite3
import cv2
import argparse

def main(args):
	db = sqlite3.connect(args.database)
	while(True):
		print("Select 1 for label search.\nSelect 2 for filename search.\nType 'quit' to exit.")
		choice = input()
		if choice == '1':
			label_search(db, args.num_results, args.output)
		elif choice == '2':
			fname_search(db, args.num_results, args.output)
		elif choice == "q" or choice == "quit":
			break
		else:
			print("Invalid input")

def label_search(db, n, o):
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
	while(True):
		l = input("Enter label to search for.\nType 'back' to return.\n")
		if l == "back":
			break
		elif l not in class_names:
			print("Invalid label.")
		else:
			print("Finding top " + str(n) + " images with label " + l)
			files = get_files_with_label(db, l, n)
			for f, r in files:
				print("File " + f + " has relevance of " + str(r))
				if o:
					if f.split('.')[-1] == 'jpg':
						img = cv2.imread('./images/' + f, 1)
						cv2.imshow(l + ' relevance = ' + str(r), img)
						cv2.waitKey(0)
					else:
						vid = cv2.VideoCapture('./videos/'+ f)
						while(vid.isOpened()):
							ret, frame = vid.read()
							cv2.imshow('frame', frame)
							if cv2.waitKey(1) & 0xFF == ord('q'):
								break
						vid.release()
				cv2.destroyAllWindows()

def get_files_with_label(db, l, n):
	c = db.cursor()
	query = "select f.filename, fl.relevance  from files f inner join file_label fl on f.file_id = fl.file_id inner join labels l on fl.label_id = l.label_id where l.label = '" + l + "' order by fl.relevance desc limit " + str(n)
	c.execute(query)
	files = c.fetchall()
	return files

def fname_search(db, n, o):
	while(True):
		f = input("Enter filename to search for.\nType 'back' to return.\n")
		if f == "back" or f == 'b':
			break
		else:
			print("Finding top " + str(n) + " labels for file " + f)
			labels = get_labels_for_file(db, f, n)
			print(labels)
			for l, r in labels:
				print("Label " + l + " has relevance of " + str(r))

			if o:
				if f.split('.')[-1] == 'jpg':
					img = cv2.imread('./images/' + f, 1)
					cv2.imshow('Image', img)
					cv2.waitKey(0)
				else:
					vid = cv2.VideoCapture('./videos/'+ f)
					while(vid.isOpened()):
						ret, frame = vid.read()
						cv2.imshow('frame', frame)
						if cv2.waitKey(1) & 0xFF == ord('q'):
							break
					vid.release()
				cv2.destroyAllWindows()
	
def get_labels_for_file(db, f, n):
	c = db.cursor()
	query = "select l.label, fl.relevance from labels l inner join file_label fl on l.label_id = fl.label_id inner join files f on fl.file_id = f.file_id  where f.filename = '" + f + "' order by fl.relevance desc limit " + str(n)
	print(query)
	c.execute(query)
	labels = c.fetchall()
	return labels

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('-db', "--database", type=str, default="labeldb.db")
	argparser.add_argument('-n', '--num_results', type=int, default=1)
	argparser.add_argument('-o', '--output', action='save_true')

	main(argparser.parse_args())