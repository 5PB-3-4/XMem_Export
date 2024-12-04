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
|cuda|11.8|
|python|3.10.15|
|uv|0.4.20|
|torch|2.3.1|
|torchvision|0.18.1|

sample mask file: ```test-sample1-1frame-mask.png``` is made by [EfficientSAM](https://github.com/opencv/opencv_zoo/tree/main/models/image_segmentation_efficientsam)


<br><br>


## ▼ Get Started
### Get Code
```shell
git clone https://github.com/5PB-3-4/XMem_Export.git -b nightly
```
<br>

### Check Dependency Library
Check out [requirement.txt](https://github.com/5PB-3-4/XMem_Export/blob/main/requirements.txt).

<br>

### Download XMem Checkpoint File
Original pretrained model is [here](https://github.com/hkchengrex/XMem/releases/tag/v1.0).

> Tested XMem-s012.pth, XMem-with-mose.pth, XMem-no-sensory.pth


<br>

### Run Export
```shell
# Run
cd XMem_Export
python exporter.py -i ./ckpt/XMem-s012.pth

# Parser option
python exporter.py -h
```

<br><br>


## ▼ ONNX Inference Code
https://github.com/5PB-3-4/XMem_ONNX/tree/main

