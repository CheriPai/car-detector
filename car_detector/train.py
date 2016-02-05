import car_detector.config as cfg
from glob import glob
from os import path
from sklearn import svm
from sklearn.externals import joblib


X = []
y = []

print("Loading positive data")
for feat_path in glob(path.join(cfg.pos_fd_path, '*.pkl')):
    fd = joblib.load(feat_path)
    X.append(fd)
    y.append(1)


print("Loading negative data")
for feat_path in glob(path.join(cfg.neg_fd_path, '*.pkl')):
    fd = joblib.load(feat_path)
    X.append(fd)
    y.append(0)

# Initalize classifier as linear SVM and train on data
print("Training classifier")
clf = svm.LinearSVC()
clf.fit(X, y)

# Serialize model to disk
joblib.dump(clf, cfg.model_path)
print("Classifier saved in {}".format(cfg.model_path))
