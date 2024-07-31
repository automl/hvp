import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import shutil

import vision_transformer as vits


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()
    
    N = 1
    mask = mask[None, :, :]
    colors = random_colors(N)
    
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask, (10, 10))
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument(
        '--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).'
        )
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument(
        '--pretrained_weights1', default='', type=str,
        help="Path to pretrained weights for model 1."
        )
    parser.add_argument(
        '--pretrained_weights2', default='', type=str,
        help="Path to pretrained weights for model 2."
        )
    parser.add_argument(
        "--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")'
        )
    parser.add_argument(
        "--image_dir", default=None, type=str,
        help="Path to the directory containing multiple JPEG images."
        )
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="We visualize masks obtained by thresholding the self-attention maps to keep xx% of the mass."
        )
    args = parser.parse_args()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model 1
    model1_name = os.path.basename(os.path.dirname(args.pretrained_weights1))
    model1 = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model1.parameters():
        p.requires_grad = False
    state_dict1 = torch.load(args.pretrained_weights1, map_location="cpu")
    state_dict1 = {k.replace("module.", ""): v for k, v in state_dict1.items()}
    model1.load_state_dict(state_dict1, strict=False)
    model1.eval().to(device)
    
    # Load model 2
    model2_name = os.path.basename(os.path.dirname(args.pretrained_weights2))
    model2 = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model2.parameters():
        p.requires_grad = False
    state_dict2 = torch.load(args.pretrained_weights2, map_location="cpu")
    state_dict2 = {k.replace("module.", ""): v for k, v in state_dict2.items()}
    model2.load_state_dict(state_dict2, strict=False)
    model2.eval().to(device)
    
    # Process each image in the directory
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(".JPEG")][:5]  # Change 5 to the desired number of images
    for image_file in image_files:
        image_path = os.path.join(args.image_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        
        transform = pth_transforms.Compose(
            [
                pth_transforms.Resize(args.image_size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        img = transform(img)
        
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)
        
        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        
        # Get attentions for model 1
        attentions1 = model1.get_last_selfattention(img.to(device))
        nh = attentions1.shape[1]
        attentions1 = attentions1[0, :, 0, 1:].reshape(nh, -1)
        
        if args.threshold is not None:
            val1, idx1 = torch.sort(attentions1)
            val1 /= torch.sum(val1, dim=1, keepdim=True)
            cumval1 = torch.cumsum(val1, dim=1)
            th_attn1 = cumval1 > (1 - args.threshold)
            idx2 = torch.argsort(idx1)
            for head in range(nh):
                th_attn1[head] = th_attn1[head][idx2[head]]
            th_attn1 = th_attn1.reshape(nh, w_featmap, h_featmap).float()
            th_attn1 = nn.functional.interpolate(
                th_attn1.unsqueeze(0), scale_factor=args.patch_size,
                mode="nearest"
                )[0].cpu().numpy()
        
        attentions1 = attentions1.reshape(nh, w_featmap, h_featmap)
        attentions1 = nn.functional.interpolate(
            attentions1.unsqueeze(0), scale_factor=args.patch_size,
            mode="nearest"
            )[0].cpu().numpy()
        
        # Get attentions for model 2
        attentions2 = model2.get_last_selfattention(img.to(device))
        attentions2 = attentions2[0, :, 0, 1:].reshape(nh, -1)
        
        if args.threshold is not None:
            val2, idx2 = torch.sort(attentions2)
            val2 /= torch.sum(val2, dim=1, keepdim=True)
            cumval2 = torch.cumsum(val2, dim=1)
            th_attn2 = cumval2 > (1 - args.threshold)
            idx2 = torch.argsort(idx2)
            for head in range(nh):
                th_attn2[head] = th_attn2[head][idx2[head]]
            th_attn2 = th_attn2.reshape(nh, w_featmap, h_featmap).float()
            th_attn2 = nn.functional.interpolate(
                th_attn2.unsqueeze(0), scale_factor=args.patch_size,
                mode="nearest"
                )[0].cpu().numpy()
        
        attentions2 = attentions2.reshape(nh, w_featmap, h_featmap)
        attentions2 = nn.functional.interpolate(
            attentions2.unsqueeze(0), scale_factor=args.patch_size,
            mode="nearest"
            )[0].cpu().numpy()
        
        # Save attentions heatmaps for model 1
        attn1_list = []
        for j in range(nh):
            fname = os.path.join(
                args.output_dir, f"{model1_name}_vs_{model2_name}",
                f"{model1_name}_attn-head{str(j)}_{image_file.replace('.JPEG', '')}.png"
                )
            os.makedirs(os.path.dirname(fname), exist_ok=True)  # Create the necessary directory
            # plt.imsave(fname=fname, arr=attentions1[j], format='png')
            # print(f"{fname} saved.")
            attn1_list.append(attentions1[j])
        
        # Save attentions heatmaps for model 2
        attn2_list = []
        for j in range(nh):
            fname = os.path.join(
                args.output_dir, f"{model1_name}_vs_{model2_name}",
                f"{model2_name}_attn-head{str(j)}_{image_file.replace('.JPEG', '')}.png"
                )
            # plt.imsave(fname=fname, arr=attentions2[j], format='png')
            # print(f"{fname} saved.")
            attn2_list.append(attentions2[j])
        
        # Convert the original image to a NumPy array
        original_image = np.asarray(img)
        
        # Create a grid image using Matplotlib subplots with adjusted spacing
        fig, axes = plt.subplots(
            nrows=2, ncols=nh + 1, figsize=(20, 8), gridspec_kw={
                'hspace':       0.1
                }
            )
    
        # Display the original image on the very left
        axes[0, 0].imshow(img[0].permute(1, 2, 0))
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Display the full model names centered above the rows with adjusted positioning
        fig.text(0.5, 0.88, model1_name, ha='center', fontsize='large', fontweight='bold')
        fig.text(0.5, 0.48, model2_name, ha='center', fontsize='large', fontweight='bold')
        
        # Display attention maps for model 1 in the first row
        for j in range(nh):
            axes[0, j+1].imshow(attentions1[j], cmap='viridis')
            axes[0, j+1].set_title(f'Head {j}')
            axes[0, j+1].axis('off')
        
        # Display attention maps for model 2 in the second row
        for j in range(nh):
            axes[1, j+1].imshow(attentions2[j], cmap='viridis')
            axes[1, j+1].set_title(f'Head {j}')
            axes[1, j+1].axis('off')
        
        # Save the grid image with adjusted spacing
        grid_fname = os.path.join(
            args.output_dir, f"{model1_name}_vs_{model2_name}",
            f"{image_file.replace('.JPEG', '')}_grid.png"
            )
        plt.savefig(grid_fname, bbox_inches='tight', pad_inches=0.1)
        print(f"{grid_fname} saved.")
        
        # Copy the original image to the output directory
        target_dir = os.path.join(args.output_dir, f"{model1_name}_vs_{model2_name}")
        shutil.copy(image_path, target_dir)
        print(f"{image_path} copied to {target_dir}.")
