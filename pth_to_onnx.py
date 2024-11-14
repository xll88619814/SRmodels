import torchvision
import torch.onnx
import torch.nn as nn
import USRNet_arch2 as USRNet_arch2
import cv2
import numpy as np
import os

def rgb():
    model = USRNet_arch2.USRNET(nf=16)
    pthfile = './BIG_RGB_v3.pth'
    model.load_state_dict(torch.load(pthfile), strict=True)
    
    lr = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, (lr), './SR_AM1_big_v3-dynamic.onnx', verbose=True, input_names=["head_input"], opset_version=11, output_names=["output"], dynamic_axes={'head_input':[0,2,3],'output':[0,2,3]})

def yuv():
    model = middle_arch.middle()
    pthfile = './middle_yuv.pth'
    model.load_state_dict(torch.load(pthfile), strict=True)
 
    lr = cv2.imread('/mnt/data/local-disk2/IVPC/SRTest/Test/afanda_000166.png')[0:512, 0:512, :]
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YUV)/255.
    lr = torch.FloatTensor(np.transpose(lr, (2, 0, 1))).unsqueeze(0)
    nparam = torch.randn(1, 1, 1, 1)

    input_names = ["head_input", "nparam"]
    output_names = ["output"]
    onnx_filename = 'models/middle_yuv__.onnx'
    torch.onnx.export(model, (lr, nparam), onnx_filename, verbose=True, input_names=input_names, output_names=output_names)

if __name__ == "__main__":
    rgb()

