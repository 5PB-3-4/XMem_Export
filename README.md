# XMem Export

> [!WARNING]
> This repogitory is work in progress.

## ▼ What's this?
This repository converts the XMem tracker model included in [Track-Anything](https://github.com/gaomingqi/Track-Anything/tree/master/tracker) from PyTorch to ONNX.

<br>

### Tested Environment
|name|version|
|----|-------|
|os|windows 10|
|python|3.10.14|
|uv|0.3.4|
|torch|2.3.1|
|torchvision|0.18.1|
|numpy|1.26.4|
|opencv|4.10.0.84|

sample mask file: ```test-sample1-1frame-mask.png``` is made by [EfficientSAM](https://github.com/opencv/opencv_zoo/tree/main/models/image_segmentation_efficientsam)


<br><br>


## ▼ Get Started
### Get Code
```shell
git clone https://github.com/5PB-3-4/XMem_Export.git
```
<br>

### Check Dependency Library
It is recommended to install using XMem's [requirement.txt](https://github.com/hkchengrex/XMem/blob/main/requirements.txt).

<br>

### Download XMem Checkpoint File
Original pretrained model is [here](https://github.com/hkchengrex/XMem/releases/tag/v1.0).

> Tested XMem-s012.pth, XMem-with-mose.pth

<br>

### Run Export
```shell
# Run
cd XMem_Export
python export.py -i ./ckpt/XMem-s012.pth --width 640 --height 480 --mask_num 1

# Parser option
python export.py -h
```

<br><br>


## ▼ ONNX Inference Code
https://github.com/5PB-3-4/XMem_ONNX/tree/main

