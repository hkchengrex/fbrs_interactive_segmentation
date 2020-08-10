import time
import sys
import os
from os import path
from argparse import ArgumentParser

import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from isegm.inference.predictors import get_predictor

from davisinteractive.dataset import Davis
from davisinteractive import utils as d_utils
from davisinteractive.utils.scribbles import scribbles2points

from isegm.utils import vis, exp
from isegm.inference import utils
from isegm.inference.clicker import Clicker, Click

device = torch.device('cuda:0')
cfg = exp.load_config_file('config.yml', return_edict=True)
torch.set_grad_enabled(False)


parser = ArgumentParser()
parser.add_argument('--output')
args = parser.parse_args()

palette = Image.open(path.expanduser('../DAVIS/2017/trainval/Annotations/480p/blackswan/00000.png')).getpalette()
os.makedirs(args.output, exist_ok=True)

image_path = '../DAVIS/2017/trainval/JPEGImages/480p'

# data stuff
davis_dataset = Davis('../DAVIS/2017/trainval')
with open('../DAVIS/2017/trainval/ImageSets/2017/train.txt', mode='r') as f:
# with open('../DAVIS/2017/trainval/ImageSets/2017/val.txt', mode='r') as f:
    subset = set(f.read().splitlines())
subset = sorted(list(subset))

# model stuff
checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, 'resnet50_dh128_lvis')
model = utils.load_is_model(checkpoint_path, device)
brs_mode = 'f-BRS-B'

input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

for scr_idx in range(1, 4):
    for seq in subset:
        scribble = davis_dataset.load_scribble(seq, scr_idx)
        all_scr = scribble['scribbles']
        for idx, s in enumerate(all_scr):
            if len(s) != 0:
                scribble['scribbles'] = [s]
                break

        annotations = davis_dataset.load_annotations(seq, dtype=np.uint8)
        h, w = annotations.shape[1:]
        points, obj_id = scribbles2points(scribble, output_resolution=(w, h))
        # points, obj_id = scribbles2points(scribble, output_resolution=(h, w))
        all_objs = np.unique(obj_id)

        image = Image.open(path.join(image_path, seq, '%05d.jpg' % idx)).convert('RGB')
        image_np = np.array(image)
        image = input_transform(image)

        predictor = get_predictor(model, brs_mode, device, prob_thresh=0.49)
        predictor.set_input_image(image)

        for ki in all_objs:
            clicker = Clicker()
            for pi in range(len(points)):
                if obj_id[pi] == ki:
                    click = Click(is_positive=True, coords=(points[pi][2], points[pi][1]))
                    clicker.add_click(click)
                # else:
                #     click = Click(is_positive=False, coords=(points[pi][2], points[pi][1]))
                #     clicker.add_click(click)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > 0.49
            clicks_list = clicker.clicks_list

            draw = vis.draw_with_blend_and_clicks(image_np, mask=pred_mask, clicks_list=clicks_list, radius=1)
            draw = np.concatenate((draw,
                255 * pred_mask[:, :, np.newaxis].repeat(3, axis=2),
            ), axis=1)

            plt.figure(figsize=(12, 16))
            plt.imshow(draw)
            plt.show()

        # this_out_path = path.join(args.output, str(scr_idx), seq)
        # os.makedirs(this_out_path, exist_ok=True)
        # for i in range(mask.shape[0]):
        #     if mask[i].max() > 0:
        #         mask_vis = Image.fromarray(mask[i].astype(np.uint8)).convert('P')
        #         mask_vis.putpalette(palette)
        #         mask_vis.save(path.join(this_out_path, '%05d.png'%i))
