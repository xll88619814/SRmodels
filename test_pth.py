import os
import numpy as np
import cv2
import torch
import time
import USRNet_arch2 as USRNet_arch2

def main():
    ############################################################################
    model_path = './BIG_RGB_v3.pth'
    
    model = USRNet_arch2.USRNET()

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.cuda()

    ii = 0
    im = cv2.imread('./zuqiu.png')/255.
    start_time = time.time()
    with torch.no_grad():
        for i in range(100):
            img_LQ = im

            h, w, _ = img_LQ.shape
            oh = int(np.ceil(img_LQ.shape[0]/4.)*4) - img_LQ.shape[0]
            ow = int(np.ceil(img_LQ.shape[1]/4.)*4) - img_LQ.shape[1]
            img_LQ = cv2.copyMakeBorder(img_LQ, oh // 2, oh - oh // 2, ow // 2, ow - ow // 2, cv2.BORDER_REFLECT)
            imgs_in = torch.FloatTensor(np.transpose(img_LQ[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).cuda()

            output = model(imgs_in).data
        
            tensor = output.squeeze().float().cpu().clamp_(*(0, 1))  # clamp
            img_np = tensor.numpy()
            img_np = np.transpose(img_np, (1, 2, 0))[:, :, [2, 1, 0]]  # CHW to HWC, RGB to BGR
            img_np = (img_np * 255.).round().astype(np.uint8)[oh:oh+h*2, ow:ow+w*2, :]
            cv2.imwrite('output.png', img_np)
               
    end_time = time.time()
    print('finish')
    print('The processing time for each image is: {}'.format((end_time-start_time)/100))

if __name__ == '__main__':
    main()
