import car_detector.config as cfg
from car_detector.helpers import pyramid, sliding_window
from argparse import ArgumentParser
from cv2 import imread, resize
from glob import glob
from os import path
from skimage import color
from skimage.feature import hog
from sklearn.externals import joblib


ap = ArgumentParser()
ap.add_argument("-p", "--pospath", required=True, help="Positive images path")
ap.add_argument("-n", "--negpath", required=True, help="Negative images path")
ap.add_argument("-t", "--test", required=False,
    action="store_true", help="Places features into test path")
args = vars(ap.parse_args())


pos_img_path = args["pospath"]
neg_img_path = args["negpath"]


if args["test"]:
    pos_fd_path = cfg.pos_fd_test_path
    neg_fd_path = cfg.neg_fd_test_path
else:
    pos_fd_path = cfg.pos_fd_path
    neg_fd_path = cfg.neg_fd_path


print('Calculating HOG for positive examples')
for image_path in glob(path.join(pos_img_path, "*")):
    image = imread(image_path, 0)
    image = resize(image, (cfg.win_width, cfg.win_height)) 
    fd = hog(image, cfg.orientations, cfg.pixels_per_cell, 
             cfg.cells_per_block, cfg.visualise, cfg.normalise)
    fd_name = path.split(image_path)[1].split(".")[0] + ".pkl"
    fd_path = path.join(pos_fd_path, fd_name)
    joblib.dump(fd, fd_path)


print('Calculating HOG for negative examples')
for image_path in glob(path.join(neg_img_path, "*")):
    image = imread(image_path, 0)
    image = resize(image, (cfg.win_width, cfg.win_height)) 
    fd = hog(image, cfg.orientations, cfg.pixels_per_cell, 
             cfg.cells_per_block, cfg.visualise, cfg.normalise)
    fd_name = path.split(image_path)[1].split(".")[0] + ".pkl"
    fd_path = path.join(neg_fd_path, fd_name)
    joblib.dump(fd, fd_path)
