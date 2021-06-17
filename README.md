# <img src="images/lvis_icon.svg" height="40"> LVIS API


LVIS (pronounced ‘el-vis’): is a new dataset for Large Vocabulary Instance Segmentation.
When complete, it will feature more than 2 million high-quality instance segmentation masks for over 1200 entry-level object categories in 164k images. The LVIS API enables reading and interacting with annotation files, visualizing annotations, and evaluating results.

<img src="images/examples.png"/>

## LVIS challenge 2021
For this release, we replace the old COCO-style Mask AP with the combination of two new metrics: [Boundary AP](https://arxiv.org/abs/2103.16562) and [Fixed AP](https://arxiv.org/abs/2102.01066). The new metric will be used in the LVIS Challenge to be held at LVIS Workshop at ICCV 2021.

## LVIS v1.0

For this release, we have annotated 159,623 images (100k train, 20k val, 20k test-dev, 20k test-challenge). Release v1.0 is publicly available at [LVIS website](http://www.lvisdataset.org) and will be used in the second LVIS Challenge to be held at Joint COCO and LVIS Workshop at ECCV 2020.

## Setup
You can setup a virtual environment and then install `lvisapi` using pip:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment

# install COCO API. COCO API requires numpy to install. Ensure that you installed numpy.
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# install OpenCV (optional if you already have OpenCV installed)
pip install -U opencv-python
# install LVIS API
pip install lvis
# Work for a while ...
deactivate  # Exit virtual environment
```

You can also clone the repo first and then do the following steps inside the repo:
```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment

# install COCO API. COCO API requires numpy to install. Ensure that you installed numpy.
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# install OpenCV (optional if you already have OpenCV installed)
pip install -U opencv-python
# install LVIS API
pip install .
# test if the installation was correct
python test.py
# Work for a while ...
deactivate  # Exit virtual environment
```
## Citing LVIS

If you find this code/data useful in your research then please cite our [paper](https://arxiv.org/abs/1908.03195):
```
@inproceedings{gupta2019lvis,
  title={{LVIS}: A Dataset for Large Vocabulary Instance Segmentation},
  author={Gupta, Agrim and Dollar, Piotr and Girshick, Ross},
  booktitle={Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
## Citing Boundary AP

If you find the Boundary AP metric useful in your research then please cite our [paper](https://arxiv.org/abs/2103.16562):
```BibTeX
@inproceedings{cheng2021boundary,
  title={Boundary {IoU}: Improving Object-Centric Image Segmentation Evaluation},
  author={Bowen Cheng and Ross Girshick and Piotr Doll{\'a}r and Alexander C. Berg and Alexander Kirillov},
  booktitle={CVPR},
  year={2021}
}
```
## Citing Fixed AP

If you find the Fixed AP metric useful in your research then please cite our [paper](https://arxiv.org/abs/2102.01066):
```BibTeX
@article{dave2021evaluating,
  title={Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details},
  author={Dave, Achal and Doll{\'a}r, Piotr and Ramanan, Deva and Kirillov, Alexander and Girshick, Ross},
  journal={arXiv preprint arXiv:2102.01066},
  year={2021}
}
```

## Credit

The code is a re-write of PythonAPI for [COCO](https://github.com/cocodataset/cocoapi).
The core functionality is the same with LVIS specific changes.  
