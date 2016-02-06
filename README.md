#car-detector
Uses histogram of oriented gradients as a descriptor and a linear support vector machine as a classifier.

Uses non-maximum suppression to eliminate overlapping bounding boxes.

#####Sample
![](data/sample/test-101.jpg?raw=false "Original Image")
![](data/sample/test-101_boxed.jpg?raw=false "Boxed Image")

#####Usage
```python car_detector/detect.py -i data/sample/test-101.jpg```

####Files
- [```config.py```](car_detector/config.py) contains general configuration values
- [```detect.py```](car_detector/detect.py) runs classifier and draws bounding boxes on sample image
- [```extract-features.py```](car_detector/extract-features.py) converts images into HOG descriptors and writes to disk to be loaded during training
- [```helpers.py```](car_detector/helpers.py) contains various helper functions
- [```test.py```](car_detector/test.py) tests classifier on training or test set printing precision and recall
- [```train.py```](car_detector/train.py) trains classifier on files output by extract-features

#####Limitations
- Only detects cars from profile view
- Has trouble with larger images

#####Resources
- [HOG+SVM Guide](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
- [NMS](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
- [Training data](https://cogcomp.cs.illinois.edu/Data/Car/)
