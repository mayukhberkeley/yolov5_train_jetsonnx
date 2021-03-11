# Train a yolov5 model on a NVIDIA Jetson NX

Assume Jetson is already set up

Verify cuda versions (10.2), driver versions

Run this to check what version of jetpack you are running

`dpkg-query --show nvidia-l4t-core`


What docker image to use in jetson, we are going to use pytorch

Using tools like robolflow for annotation, we can also use labelImage

*tegrastats

*top

*tensorflow

train for more epochs

the right version of torch is important

many issues with the folder structure of what we get from Yolov5 repo

opencv installation is an issue on jetson, is it really needed for Yolo?

Do I already have an opencv installed, I think I do, that it really may be needed.

Building python-opencv version 4.1.2, this version is not available for install for aarch64

`apt install -y qt4-default`

`git clone https://github.com/opencv/opencv-python.git`

`cd opencv-python`

`git checkout -b 412 tags/30`

`python3 setup.py bdist_wheel`

`pip3 install <>.whl`


--runtime=nvidia is needed for python3 to find torch during import torch

Fix the matplotlib issue, check is there is already no matplotlib 2.1.1 from apt

`sudo docker run --rm -it --runtime=nvidia --name pytorchcoco --shm-size=1G -v ~/w251/finalproject/app:/app -p 8888:8888 -p 6006:6006 pytorchcoco`

train for more epochs, check the precision and recall to make sure we are training enough, 
keep and eye on the device memory usage, check for leaks

import order is important when running in a notebook

import cv2
import torch

if you still see issues with importing torch, e.g. unable to locate 'six' etc, restart the Jupyter kernel
NOTE: import torch works find in a python3 shell, just issues with the notebook


pytorch 1.7.0 release is not available for install via pip for aacrh64, we will need to build it from source

`git clone https://github.com/pytorch/pytorch.git`

`cd pytorch`

`git checkout -b torch170 tags/v1.7.0`

`git submodule update --init --recursive`

`python3 setup.py bdist_wheel`

`pip3 install <>.whl`


