#car-detector
![](data/sample/test-101.jpg?raw=false "Original Image")
![](data/sample/test-101_boxed.jpg?raw=false "Boxed Image")

Uses histogram of oriented gradients as a descriptor and a linear support vector machine as a classifier.

Uses non-maximum suppression to eliminate overlapping bounding boxes.


#####Usage
```python car_detector/detect.py -i data/sample/test-101.jpg```

#####Limitations
- Only detects cars from profile view
- Has trouble with larger images

#####Resources
- [HOG+SVM Guide](http://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
- [NMS](http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)
- [Training data](https://cogcomp.cs.illinois.edu/Data/Car/)
