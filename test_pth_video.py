import os
import os.path as osp
import glob
import numpy as np
import cv2
import torch
import time
import USRNet_arch2 as USRNet_arch2

def main():
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    ############################################################################
    model_path = './BIG_RGB_v3.pth'
    
    model = USRNet_arch2.USRNET()
    video_path = './captain.mp4'
    cap = cv2.VideoCapture(video_path)

    save_folder = './results_pth/'
    save_subfolder = osp.join(save_folder, 'JYX-W10_quality')
    if not os.path.exists(save_subfolder):
        os.makedirs(save_subfolder)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    ii = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if ret:
                if True:
                    img_name = str(ii).zfill(6)
                    img_LQ = frame/255.

                    h, w, _ = img_LQ.shape
                    oh = int(np.ceil(img_LQ.shape[0]/4.)*4) - img_LQ.shape[0]
                    ow = int(np.ceil(img_LQ.shape[1]/4.)*4) - img_LQ.shape[1]
                    img_LQ = cv2.copyMakeBorder(img_LQ, oh // 2, oh - oh // 2, ow // 2, ow - ow // 2, cv2.BORDER_REFLECT)
                    imgs_in = torch.FloatTensor(np.transpose(img_LQ[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)

                    output = model(imgs_in).data
        
                    tensor = output.squeeze().float().cpu().clamp_(*(0, 1))  # clamp
                    img_np = tensor.numpy()
                    img_np = np.transpose(img_np, (1, 2, 0))[:, :, [2, 1, 0]]  # CHW to HWC, RGB to BGR
                    img_np = (img_np * 255.).round().astype(np.uint8)[oh:oh+h*2, ow:ow+w*2, :]

                    cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), img_np)
                ii += 1
            else:
                break
    print('finish')

if __name__ == '__main__':
    main()
