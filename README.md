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

Pre build instructions

https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html


`apt install -y qt4-default`

`git clone https://github.com/opencv/opencv-python.git`

`cd opencv-python`

`git checkout -b 412 tags/30`

`python3 setup.py bdist_wheel`

`pip3 install <>.whl`


--runtime=nvidia is needed for python3 to find torch during `import torch`

Fix the matplotlib issue, check if there is already e.g. matplotlib 2.1.1 from apt

`sudo docker run --rm -it --runtime=nvidia --name pytorchcoco --shm-size=1G -v ~/w251/finalproject/app:/app -p 8888:8888 -p 6006:6006 pytorchcoco`

train for more epochs, check the precision and recall to make sure we are training enough, 
keep and eye on the device memory usage, check for leaks

import order is important when running in a notebook

import cv2
import torch

if you still see issues with importing torch, e.g. unable to locate 'six' etc, restart the Jupyter kernel
NOTE: import torch works find in a python3 shell, just issues with the notebook

Check in pytorch NGC https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch/tags

The image compatible with R32.4.4 which is jetpack 4.4.1 does not have pytorch 1.7 or torchvision 0.8.2

pytorch 1.7.0 release is not available for install via pip for aacrh64 either, we will need to build it from source

Install Qt first...

`apt-get install -y qt4-default`

`git clone https://github.com/pytorch/pytorch.git`

`cd pytorch`

`git checkout -b torch170 tags/v1.7.0`

`git submodule update --init --recursive`

`python3 setup.py bdist_wheel`

`pip3 install <>.whl`


We also need to build torchvision v0.8.2


Tried yolov5 with latest, should try v4.0 instead



Alternative, use the NGC from nvidia for pytorch 1.6 (R32.4.4) Nvidia Jetson Xavier NX and use the wheels from here
http://mathinf.com/pytorch/arm64/2021-01/

you will need to update to python 3.7.5, these wheels are compiled for cp37

fix the link to python3

`update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1`

`update-alternatives --config python3`

`pip3 install opencv-python`

Package           Version
----------------- ---------------
numpy             1.20.1
opencv-python     4.5.1.48
Pillow            8.1.2
pip               21.0.1
torch             1.8.0a0
torchvision       0.8.0a0+45f960c
typing-extensions 3.7.4.3

`git clone https://github.com/ultralytics/

v5.git`

`cd yolov5`

`pip3 install -r requirements.txt`

You will still see issues when installing matplotlib >= 3.3.2

Install setuptools and try again

`apt-get install python3-setuptools`

`pip3 install -r requirements.txt`

This will fail again, we will need to build torchvision >= 0.8.1

Optionally download the wheel file for the torch 1.8.0 from here https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048


Building grpcio would be too slow, instead try using python 37

https://github.com/grpc/grpc/issues/20493


`apt-get install python3.8`

Fix the links 
Then, install pip for python3.8

`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

`python3.8 get-pip.py`

opencv-python-4.5.1.48 causes a code dump segmentation fault with torch 1.8.0 ( torchvision 0.9.0 )


Could be use to python 3.8, can't be for sure. reverting back to python 3.6

There seems to be no wheel for matplotlib for aarch64 an cp36, there is one for cp37 though

Need to build it from source

Need to install Cmake

Built wheel for torch vision 0.8.1

`apt install cmake`

Now install opencv-python 

This will build the wheel filep

`pip3 install jupyterlab`

`jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --notebook_dir=/app`


`sudo docker run --rm -it --runtime=nvidia --name pytorchcoco --shm-size=1G -v ~/w251/finalproject/app:/app -p 8888:8888 -p 6006:6006 pytorchcoco`


running `pip3 install -r requirements.txt` from yolov5 again

This is what you will see depending upon what you have already preinstalled

Building wheels for collected packages: matplotlib, PyYAML, scipy, pandas, pycocotools, kiwisolver, grpcio
  Running setup.py bdist_wheel for matplotlib ... -
  
_grpcio_ takes the most time, we have a wheel file prebuilt for that, apparently it is supposed to be faster with python 3.7 but we didn't try that

it will install matplotlib now, may build from source, we already have 2.1.1 that came with apt install python3-matplotlib, this will throw during import, matplotlib 3.3.4 is needed

basically, we are just getting nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

Then we will install torch 1.7 from wheel file available in nvidia and torchvision 0.8.1 will be built locally (for cp36)

opencv python will be build for version 4.5.1 locally for which we need to install CMake `apt install cmake`

Building opencv 4.5.1 when torch 1.8.0 was installed, failed, we reverted to torch 1.7

opencv 4.5 brings in numpy 1.19.5 which causes import cv2 to crash.

downgrade to 1.19.2 works

or, use opencv-python==4.4.0.46, may need to compile it

While running the Yolov5 training you may run into dateutil issues, so ensure you have the correct one

pip3 install python-dateutil==2.7.3

Again, when running training, you are in yolov5 folder since thats where train.py is
ensure the valid, train and test are in a folder ../ w.r.t. yolov5 folder


I strongly suggest using roboflow to fetch annotated images


### AWS - torch containers for the cloud 

On the cloud image, you will see issues when importing cv2
You will need to install the necessary libraries

the cloud container is based on 

FROM nvcr.io/nvidia/pytorch:21.02-py3

This already comes with opencv 4.4.0 installed but it throws when importing

Need to do the following to fix it

`apt-get update`

`apt-get install ffmpeg libsm6 libxext6  -y`
















