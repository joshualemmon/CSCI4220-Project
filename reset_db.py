import sqlite3 
import os
import argparse


def add_images(db, img_dir):
	c = db.cursor()
	for i,f in enumerate(os.listdir(img_dir)):
		c.execute("insert into files(file_id, filename) values ("+str(i)+", '" + f + "');")
	db.commit()
		
def add_videos(db, vid_dir):
	c = db.cursor()
	c.execute("select file_id from files order by file_id desc")
	latest_id = c.fetchone()[0]
	for i,f in enumerate(os.listdir(vid_dir), latest_id +1):
		c.execute("insert into files(file_id, filename) values ("+str(i)+", '" + f + "');")
	db.commit()

def add_labels(db):
	labels = []
	with open('labels.txt', 'r') as f:
		for l in f.readlines():
			labels.append(l.strip('\n'))
	c = db.cursor()
	for i, l in enumerate(labels):
		c.execute("insert into labels(label_id, label) values ("+str(i) + ", '" + l +"');")
	db.commit()

def main(args):
	db = sqlite3.connect(args.db_path)
	c = db.cursor()
	c.execute("delete from files;")
	c.execute("delete from labels;")
	if args.reset_file_labels:
		c.execute("delete from file_label;")
	db.commit()
	add_labels(db)
	add_images(db, args.img_dir)
	add_videos(db, args.vid_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-db', '--db_path', type=str, default='labeldb.db')
	parser.add_argument('-id', '--img_dir', type=str, default='./images')
	parser.add_argument('-vd', '--vid_dir', type=str, default='./videos')
	parser.add_argument('-r', '--reset_file_labels', action='store_true')
	main(parser.parse_args())