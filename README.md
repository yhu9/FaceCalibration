# FaceCalibration
Camera Self-Calibration Using Human Faces. You can train the model and create a synthetic testing data yourself or simply download the trained model and testing data.

## Creating Environment
To create the environment from scratch using a conda environment use the following steps on Ubuntu.

1. conda create -n fcc python=3.6
2. conda install pytorch pytorch=1.10.0 torchvision cudatoolkit=10.2
3. pip install scikit-image matplotlib imageio plotly opencv-python             
4. pip install pyyaml

Due to weirdness which I do not fully comprehend, make sure that you import torch before you import cv2 on any code you build on top of this repo. Otherwise you might run into GLIBCXX error depending on your machine.

# Trained Model and Training
The trained model is extremely small and is in the github repo under the model directory. If you clone the repo you should have the trained model already.

```
python train.py
```

# Testing Data
If you want to generate the synthetic testing data yourself, you'll need a 3DMM model, matlab, and run the synthetic data generation code in the matlab folder. The 3DMM model used can be downloaded at the following link. Once downloaded extract to the base of the project.


https://drive.google.com/file/d/1U_nPN6Q2x83KjTphwO7kECTnu5rEOGRV/view?usp=share_link


Otherwise download the testing data and extract to the base of the project to perform testing. 

https://drive.google.com/file/d/1dKPMalkELumb0AjgE5opsfK_ykCzOIt4/view?usp=share_link


Once installed, run testing with the command

```
python test.py
```

# Citation
http://cvlab.cse.msu.edu/pdfs/Hu_Brazil_Li_Ren_Liu_FG2023.pdf

```
@article{hucamera,
  title={Camera Self-Calibration Using Human Faces},
  author={Hu, Masa and Brazil, Garrick and Li, Nanxiang and Ren, Liu and Liu, Xiaoming}
}
```
