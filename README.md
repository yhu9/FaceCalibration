# FaceCalibration
<<<<<<< HEAD
Camera Self-Calibration Using Human Faces

## Creating Environment
Due to weirdness which I do not fully comprehend, make sure that you import torch before you import cv2 on any code. Otherwise you might run into GLIBCXX error depending on
your machine. I ran this on a remote server where I did not have admin privledges so I honestly cannot figure out exactly why this happens.

1. conda create -n fcc python=3.6
2. conda install pytorch pytorch=1.10.0 torchvision cudatoolkit=10.2
3. pip install scikit-image matplotlib imageio plotly opencv-python             
4. pip install pyyaml
=======

This github holds the codebase for the paper "Camera Self-Calibration Using Human Faces".
>>>>>>> c57286b20dd03b9d1deabd78a20ed319c6a4e0df
