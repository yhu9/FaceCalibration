# FaceCalibration
Camera Self-Calibration Using Human Faces. To run this code base you'll need the testing data, the trained model, and the 3DMM which I stil haven't gotten around uploading. In the meantime, you can make your own testing data by running the matlab code. You can also train as is since training is done on the fly.

## Creating Environment
Due to weirdness which I do not fully comprehend, make sure that you import torch before you import cv2 on any code. Otherwise you might run into GLIBCXX error depending on
your machine. I ran this on a remote server where I did not have admin privledges so I honestly cannot figure out exactly why this happens.

1. conda create -n fcc python=3.6
2. conda install pytorch pytorch=1.10.0 torchvision cudatoolkit=10.2
3. pip install scikit-image matplotlib imageio plotly opencv-python             
4. pip install pyyaml


# Citation
http://cvlab.cse.msu.edu/pdfs/Hu_Brazil_Li_Ren_Liu_FG2023.pdf

```
@article{hucamera,
  title={Camera Self-Calibration Using Human Faces},
  author={Hu, Masa and Brazil, Garrick and Li, Nanxiang and Ren, Liu and Liu, Xiaoming}
}
```
