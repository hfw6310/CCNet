import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import cv2 as cv
import torch.nn.functional as f
import numpy as np
from pytorch_msssim import ssim

def _eval(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    factor = 8

    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            
            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # label_img = (label_img).cuda()
            # down_ratio = max(1, round(min(H, W) / 256))	
            # print('down_ratio:',down_ratio)
            # # ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
            # #                 f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
            # #                 data_range=1, size_average=False)
            # ssim_val = ssim(pred_clip, label_img, data_range=1, size_average=False)
            # ssim_adder(ssim_val)

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            # print('pred_numpy.shape:',pred_numpy.shape)
            # print('np.transpose(pred_numpy,(1,2,0)).shape:',np.transpose(pred_numpy,(1,2,0)).shape)
            ssim_val = structural_similarity(cv.cvtColor(np.transpose(pred_numpy,(1,2,0)),cv.COLOR_BGR2GRAY), 
                                                cv.cvtColor(np.transpose(label_numpy,(1,2,0)),cv.COLOR_BGR2GRAY),data_range=1)
            ssim_adder(ssim_val)
            
            print('%d iter PSNR: %.5f ' % (iter_idx + 1, psnr))
            print('%d iter SSIM: %.5f ' % (iter_idx + 1, ssim_val))

        print('==========================================================')
        print('The average PSNR is %.3f dB' % (psnr_adder.average()))
        print('The average SSIM is %.3f ' % (ssim_adder.average()))

