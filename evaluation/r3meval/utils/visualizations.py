import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T

import mvp

def place_attention_heatmap_over_images(images, vision_model, head=1):

    H, W = 224, 224
    patch_size = 16
    alpha = .4
    new_H, new_W = H//patch_size, W//patch_size

    transforms = T.Compose([T.ToTensor(),
                            T.Resize(H),
                            T.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))])

    cmap = plt.get_cmap('jet')
    

    heatmap_images = []
    for image in images: # can vectorize?

        # resize and normalize image to feed into model
        image = image.copy()
        torch_image = transforms(image)

        # grab the output attention map at the desired attention head
        # attn = vision_model.forward_attention(torch_image.unsqueeze(0), layer=11)
        attn = vision_model.get_last_selfattention(torch_image.unsqueeze(0).to('cuda'))
        attn_map = attn[0,head,0,1:].reshape(1, 1, new_H, new_W) # B, C, H, W

        # interpolate smoothly to create a heatmap
        resized_attn_map = F.interpolate(attn_map, scale_factor=patch_size,
                                         mode='bilinear')
        resized_attn_map = resized_attn_map.cpu().detach().numpy().squeeze()

        # convert attention scores to heatmap
        image = cv2.resize(image, (W, H))
        heatmap = cmap(resized_attn_map/resized_attn_map.max())
        heatmap *= 255
        heatmap = heatmap[:,:,:3]
        heatmap_image = (.8*image + .2*heatmap).astype(int)
        heatmap_image = np.clip(heatmap_image, 0, 255)
        heatmap_images.append(heatmap_image)

    return heatmap_images

