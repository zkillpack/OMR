# Optical Music Recognition

This is an (unfinished) attempt to expand some musical image segmentation code I prototyped for a computer vision class in Matlab. The body of the project involves detecting the locations of staff lines and musical objects (which are composed of primitives such as stems and noteheads). The next step involves training an exemplar SVM via hard negative mining to detect different note types and then detect pitch via vertical positioning with respect to the staff grid.

Most of the algorithms in this repo are various experiments at removing staff lines, including entropy-based and dynamic programming methods.

To truly experience the uselessness, run omr.py with one of the supplied examples:
```
python omr.py -i .\Examples\f.png
```

This project requires:
* Python 2.7.8
* OpenCV 2.4.10
* Numpy 1.9.1
* Scipy 0.15.1
