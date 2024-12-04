import argparse

import torch

from base_tracker import BaseTracker
from export_model import *

parser = argparse.ArgumentParser(description='welcome to Xmem exporter v0.5')
parser.add_argument('-i', '--input_dir', required=True, help='Xmem checkpoint file path')
parser.add_argument('-o', '--output_dir', required=False, default='./export/', help='export folder path. default is [ export/ ]')
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
# print("CUDA is available:", torch.cuda.is_available())
print()



print("=================================================================================")
print("export model")
print("=================================================================================")
input_w = 640
input_h = 480
input_c = 3
w_div16 = int(input_w/16)
h_div16 = int(input_h/16)
input_m = 1

is_hidden_dim = Btrack.tracker.network.hidden_dim > 0
hidden_dim = 64 if is_hidden_dim else 0

print("is hidden_dim ? : ", is_hidden_dim)
print("set value")

print("is export model ?")
do_export_flag = { "encode_key": True, "encode_value": True, "decode": True }
for k, v in do_export_flag.items():
    print("{}:  {}".format(k, v))

print("===================================")
print("model: key encoder + key projection")
model_encode_key = EncodeKey(
    Btrack.tracker.network.key_encoder,
    Btrack.tracker.network.key_proj
    ).eval().cpu()

if do_export_flag["encode_key"]:
    print("\nexport\n")

    dummy_inputs = (
        torch.randn((1, input_c, input_h, input_w)),
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
            "image"    : { 1: "c",           2: "h",  3: "w"  },
            "key"      : { 1: "hidden_dim",  2: "y",  3: "x" },
            "shrinkage": {                   2: "y",  3: "x" },
            "selection": { 1: "hidden_dim",  2: "y",  3: "x" },
            "f16"      : { 1: "value_dim*2", 2: "y",  3: "x" },
            "f8"       : { 1: "value_dim",   2: "2y", 3: "2x"},
            "f4"       : { 1: "value_dim/2", 2: "4y", 3: "4x"}
        },
        verbose=False
    )

    print("\nexport done!\n")


print("===================================")
print("model: value encoder")
model_encode_value = EncodeValue(Btrack.tracker.network.value_encoder).eval().cpu()

if do_export_flag["encode_value"]:
    print("\nexport\n")
    
    dummy_inputs = (
        torch.randn((1, input_c, input_h, input_w)),
        torch.randn((1, 1024, h_div16, w_div16)),
        torch.randn((1, input_m, hidden_dim, h_div16, w_div16)),
        torch.randn((1, input_m, input_h, input_w)),
        torch.randn((1, input_m, input_h, input_w)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_encode_value,
        dummy_inputs,
        "./export/XMem-encode_value.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= ["image","f16", "h16_in", "masks", "others", "is_deep_update"],
        output_names=["g16", "h16_out"],
        dynamic_axes={
            "image"  : { 1: "c",           2: "h",         3: "w"         },
            "f16"    : { 1: "value_dim*2", 2: "y",         3: "x"         },
            "h16_in" : { 1: "input_m",    2: "hidden_dim", 3: "y", 4: "x" },
            "masks"  : { 1: "input_m",    2: "y",          3: "x"         },
            "others" : { 1: "input_m",    2: "y",          3: "x"         },
            "g16"    : { 1: "input_m",    2: "value_dim",  3: "y", 4: "x" },
            "h16_out": { 1: "input_m",    2: "hidden_dim", 3: "y", 4: "x" },
        },
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
        torch.randn((1, input_m, hidden_dim, h_div16, w_div16)),
        torch.randn((1, input_m, 512, h_div16, w_div16)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_segment,
        dummy_inputs,
        "./export/XMem-decode.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=False,
        input_names= ["f16","f8","f4", "h16_in", "memory_readout", "h_out"],
        output_names=["h16_out", "logits", "prob"],
        dynamic_axes={
            "f16"           : { 1: "value_dim*2", 2: "y",          3: "x"         },
            "f8"            : { 1: "value_dim",   2: "2y",         3: "2x"        },
            "f4"            : { 1: "value_dim/2", 2: "4y",         3: "4x"        },
            "h16_in"        : { 1: "input_m",     2: "hidden_dim", 3: "y", 4: "x" },
            "memory_readout": { 1: "input_m",     2: "value_dim",  3: "y", 4: "x" },
            "h16_out"       : { 1: "input_m",     2: "hidden_dim", 3: "y", 4: "x" },
            "logits"        : { 1: "input_m",     2: "h",          3: "w"         },
            "prob"          : { 1: "input_m",     2: "h",          3: "w"         }
        },
  verbose=False
    )

    print("\nexport done!\n")


print("=================================================================================")
