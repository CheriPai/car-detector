import car_detector.config as cfg
from car_detector.helpers import pyramid, sliding_window, non_max_suppression
from argparse import ArgumentParser
from cv2 import imread, imwrite, rectangle, waitKey
from os import path
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.externals import joblib
from sys import exit


ap = ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path")
args = vars(ap.parse_args())

boxes = []
image = imread(args["image"])
if image is None:
    print("Error: No image found at '{}'".format(args["image"]))
    exit()

# Load previously trained classifier
try:
    clf = joblib.load(cfg.model_path)
except FileNotFoundError:
    print("Error: Model not found at '{}'".format(cfg.model_path))
    print("Check the path in config.py or run train.py to generate a model")
    exit()


# Slides window across image running HOG and classifier on each window
for i, resized in enumerate(pyramid(color.rgb2gray(image), cfg.scale_amt)):
    for (x, y, window) in sliding_window(resized, cfg.step_size, 
                            window_size=(cfg.win_width, cfg.win_height)):
        if window.shape[0] != cfg.win_height or window.shape[1] != cfg.win_width:
            continue

        fd = hog(window, cfg.orientations, cfg.pixels_per_cell, 
                 cfg.cells_per_block, cfg.visualise, cfg.normalise)
        if clf.predict(fd.reshape(1, -1)) == 1:
            boxes.append((x, y, x+cfg.win_width*cfg.scale_amt**i,
                y+cfg.win_height*cfg.scale_amt**i))


# Remove redundant boxes
boxes = non_max_suppression(boxes, cfg.overlap_thresh)
for (x1, y1, x2, y2) in boxes:
    rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


old_path = path.split(args["image"])[1].split(".")
new_path = old_path[0] + "_boxed." + old_path[1]
imwrite(new_path, image)
print("Saved as {}".format(new_path))
