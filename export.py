import sys
import argparse
import numpy as np
import cv2
import torch

from base_tracker import BaseTracker
from export_model import *


parser = argparse.ArgumentParser(description='welcome to Xmem exporter v0.3')
parser.add_argument('-i', '--input_dir', required=True, help='Xmem checkpoint file path')
parser.add_argument('-o', '--output_dir', required=False, default='./export/', help='export folder path. default is [ export/ ]')
parser.add_argument('--width', required=False, default=640, help='model input width. default is [ 640 ] ')
parser.add_argument('--height', required=False, default=480, help='model input height. default is [ 480 ] ')
parser.add_argument('--mask_num', required=False, default=1, help='the number of mask. default is [ 1 ] ')
args = parser.parse_args()
print()


print("=================================================================================")
print('initialize model')
print("=================================================================================")
xmem_checkpoint = args.input_dir
device = "cpu"
Btrack = BaseTracker(xmem_checkpoint, device)
Btrack.clear_memory()

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
print()


print("=================================================================================")
print("set dummy inputs")
print("=================================================================================")
# set padding image shape
def pad_divide_by(in_img, d=16):
    h, w = in_img.shape[0], in_img.shape[1]
    new_w, new_h = 0, 0
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))  # (padding_left,padding_right,padding_top,padding_bottom)
    pad_val = None
    if len(in_img.shape) == 3:
        pad_val = (0, 0, 0)
    elif len(in_img.shape) == 2:
        pad_val = 0
    # out = cv2.resize(in_img, dsize=(new_w, new_h))
    out = cv2.copyMakeBorder(in_img, top=lh,bottom=uh,left=lw,right=uw,borderType=cv2.BORDER_CONSTANT, value=pad_val)
    return np.array(out,dtype=np.float32), pad_array

w_set  = abs(int(args.width))
h_set  = abs(int(args.height))
dummy_img = np.zeros((h_set, w_set, 3))
print("setting dummy image shape: w={}, h={}".format(w_set, h_set))

dummy_img, _ = pad_divide_by(dummy_img)
print("padding dummy image shape: w={}, h={}".format(dummy_img.shape[1], dummy_img.shape[0]))
h_new, w_new, ch_new = dummy_img.shape
h_pad  = int(h_new)
w_pad  = int(w_new)
ch_pad = 3
mask_num = int(args.mask_num)
mask_num = max(1, min(255, mask_num))


print("=================================================================================")
print("export model")
print("=================================================================================")
w_div16 = int(w_pad/16)
h_div16 = int(h_pad/16)

do_export_flag = { "encode_key": True, "encode_value": True, "decode": True }
for k, v in do_export_flag.items():
    print("{}:  {}".format(k, v))

print("\nset value")

print("===================================")
print("model: key encoder + key projection")
model_encode_key = EncodeKey(
    Btrack.tracker.network.key_encoder,
    Btrack.tracker.network.key_proj
    ).eval().cpu()

if do_export_flag["encode_key"]:
    print("\nexport\n")

    dummy_inputs = (
        torch.randn((1, ch_pad, h_pad, w_pad)),
        torch.randn((1)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_encode_key,
        dummy_inputs,
        "./export/XMem-encode_key.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names= ["image", "need_sk", "need_ek"],
        output_names=["key", "shrinkage", "selection", "f16", "f8", "f4"],
        dynamic_axes={
            "image"    : { 2: "h",  3: "w" },
            "key"      : { 2: "x",  3: "y" },
            "shrinkage": { 2: "x",  3: "y" },
            "selection": { 2: "x",  3: "y" },
            "f16"      : { 2: "x",  3: "y" },
            "f8"       : { 2: "x2", 3: "y2"},
            "f4"       : { 2: "x4", 3: "y4"}
        },
        verbose=False
    )

    print("\nexport done!\n")


print("===================================")
print("model: value encoder")
model_encode_value = EncodeValue(Btrack.tracker.network.value_encoder).eval().cpu()
is_hidden_dim = model_encode_value.is_hidden_dim.to('cpu').detach().numpy().copy()[0]
hidden_dim = 64 if is_hidden_dim else 0

if do_export_flag["encode_value"]:
    print("\nexport\n")
    
    dummy_inputs = (
        torch.randn((1, ch_new, h_new, w_new)),
        torch.randn((1, 1024, h_div16, w_div16)),
        torch.randn((1, mask_num, hidden_dim, h_div16, w_div16)),
        torch.randn((1, mask_num, h_new, w_new)),
        torch.randn((1, mask_num, h_new, w_new)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_encode_value,
        dummy_inputs,
        f"./export/XMem-encode_value-m{mask_num}.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= ["image","f16", "h16_in", "masks", "others", "is_deep_update"],
        output_names=["g16", "h16_out"],
        verbose=False
    )

    print("\nexport done!\n")


print("===================================")
print("model: decoder")
model_segment = Segment(
    Btrack.tracker.network.decoder,
    Btrack.tracker.network.value_dim
    ).eval().cpu()

if do_export_flag["decode"]:
    print("\nexport\n")

    dummy_inputs = (
        torch.randn((1, 1024, h_div16, w_div16)),
        torch.randn((1, 512, int(2*h_div16), int(2*w_div16))),
        torch.randn((1, 256, int(4*h_div16), int(4*w_div16))),
        torch.randn((1, mask_num, hidden_dim, h_div16, w_div16)),
        torch.randn((1, mask_num, 512, h_div16, w_div16)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_segment,
        dummy_inputs,
        f"./export/XMem-decode-m{mask_num}.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names= ["f16","f8","f4", "h16_in", "memory_readout", "h_out"],
        output_names=["h16_out", "logits", "prob"],
        verbose=False
    )

    print("\nexport done!\n")


print("=================================================================================")
