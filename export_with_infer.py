import warnings
import argparse
import numpy as np
import cv2
import torch

from base_tracker import BaseTracker
from export_model import *


parser = argparse.ArgumentParser(description='welcome to Xmem exporter v0.2')
parser.add_argument('-i', '--input_dir', required=True, help='Xmem checkpoint file path')
parser.add_argument('-o', '--output_dir', required=False, default='./export/', help='export folder path. default is [ export/ ]')
parser.add_argument('--width', required=False, default=640, help='model input width. default is [ 640 ] ')
parser.add_argument('--height', required=False, default=480, help='model input height. default is [ 480 ] ')
parser.add_argument('--mask_num', required=False, default=1, help='max number of mask. default is [ 1 ] ')
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

# read sample video
video_path = './sample/test-sample1.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("please set sample video file path")
    raise IOError
fn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# read mask image
masks = cv2.imread('sample/test-sample1-1frame-mask.png', cv2.IMREAD_GRAYSCALE)
print("sample image shape: w={}, h={}".format(masks.shape[1], masks.shape[0]))
masks = cv2.resize(masks, dsize=(w_set, h_set))

# color clustering
mask_num = int(args.mask_num)
mask_num = max(1, min(256, mask_num+1))
best_mask = np.zeros_like(masks)
if 1 < mask_num:
    masks = np.floor(masks/(256.0/mask_num))*256.0/mask_num
    best_mask = masks.astype(np.uint8)

    prelabels = np.unique(best_mask).astype(np.uint8)
    prelabels = prelabels[prelabels!=0].tolist()
    if (len(prelabels)) != mask_num:
        warnings.warn("The number of masks did not match the number specified. \nThe number of masks will be Re-specified: {}".format(len(prelabels)))

else:
    _, best_mask = cv2.threshold(masks, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    warnings.warn("The number of masks did not match the number specified. \nThe number of masks will be Re-specified: 1")

best_mask = best_mask.astype(np.uint8)

print()


print("=================================================================================")
print("infer test")
print("=================================================================================")
cv2.namedWindow('result', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("result", w_pad, h_pad)
first_frame = True
print("sample eval start.")
try_count = 10 if fn>10 else fn
for i in range(try_count):
    try:
        ret, frame = cap.read()
        if ret is False:
            raise IOError
        frame = cv2.resize(frame, dsize=(w_set, h_set))
        print("===================================")
        print("frame #" + str(i))

        if first_frame:
            mask, prob, painted_frame = Btrack.track(frame, best_mask)
            first_frame = False
        else:
            mask, prob, painted_frame = Btrack.track(frame)
        
        best_mask = cv2.normalize(src=mask, dst=None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        best_mask = cv2.convertScaleAbs(best_mask, dst=None, alpha=255.0, beta=0.0)

        cvtImg = np.hstack((frame, cv2.cvtColor(best_mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("result", cv2.resize(cvtImg, dsize=None, fx=0.5, fy=0.5))
        cv2.waitKey(1)
    except KeyboardInterrupt:
        break

cv2.destroyAllWindows()
cap.release()

print("===================================")
print("sample eval end.\n")


print("=================================================================================")
print("export model")
print("=================================================================================")
# get value
key, shrinkage, selection, f16, f8, f4 = Btrack.tracker.export_val_enc_key
memory_readout, hidden_1, logits, prob = Btrack.tracker.export_val_dec
value, hidden_2, all_labels = Btrack.tracker.export_val_enc_val

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
        verbose=False
    )

    print("\nexport done!\n")


print("===================================")
print("model: value encoder")
model_encode_value = EncodeValue(Btrack.tracker.network.value_encoder).eval().cpu()

if do_export_flag["encode_value"]:
    print("\nexport\n")

    dummy_inputs = (
        torch.randn((1, ch_new, h_new, w_new)),
        torch.randn_like(f16),
        torch.randn_like(hidden_2),
        torch.randn_like(prob[0, 1:].unsqueeze(0)),
        torch.randn_like(prob[0, 1:].unsqueeze(0)),
        torch.randn((1))
    )

    torch.onnx.export(
        model_encode_value,
        dummy_inputs,
        f"./export/XMem-encode_value.onnx",
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
        torch.randn_like(f16),
        torch.randn_like(f8),
        torch.randn_like(f4),
        torch.randn_like(hidden_1),
        torch.randn_like(memory_readout),
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
        verbose=False
    )

    print("\nexport done!\n")


print("=================================================================================")
