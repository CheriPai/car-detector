import car_detector.config as cfg
from argparse import ArgumentParser
from glob import glob
from os import path
from sklearn.externals import joblib


ap = ArgumentParser()
ap.add_argument("-tr", "--train", required=False,
    action="store_true", help="Places features into test path")
ap.add_argument("-hn", "--hard-negative", required=False,
    action="store_true", help="Copies false negatives into neg_fd_path")
args = vars(ap.parse_args())


if args["train"]:
    print("Testing on training set")
    pos_fd_path = cfg.pos_fd_path
    neg_fd_path = cfg.neg_fd_path
else:
    print("Testing on testing set")
    pos_fd_path = cfg.pos_fd_test_path
    neg_fd_path = cfg.neg_fd_test_path


num_pos = 0
pos_correct = 0
num_neg = 0
neg_correct = 0


# Initalize classifier
try:
    clf = joblib.load(cfg.model_path)
except FileNotFoundError:
    print("Error: Model not found at '{}'".format(cfg.model_path))
    print("Check the path in config.py or run train.py to generate a model")
    exit()


for fd_path in glob(path.join(pos_fd_path, '*.pkl')):
    num_pos += 1
    fd = joblib.load(fd_path)
    if clf.predict(fd.reshape(1, -1)) == 1:
        pos_correct += 1
        

for fd_path in glob(path.join(neg_fd_path, '*.pkl')):
    num_neg += 1
    fd = joblib.load(fd_path)
    if clf.predict(fd.reshape(1, -1)) == 0:
        neg_correct += 1
    elif "train" in args and "hard-negative" in args:
        # Perform hard negative mining
        fd_path = fd_path.split(".")[0] + "_HN.pkl"
        joblib.dump(fd, fd_path)


print("Precision: {}".format(pos_correct/(pos_correct+num_neg-neg_correct)))
print("Recall: {}".format(pos_correct/num_pos))
