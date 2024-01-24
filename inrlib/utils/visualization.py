import os
from os import path
from PIL import Image, ImageDraw
from typing import Literal, List, Callable
import torch
from matplotlib import pyplot as plt
import numpy as np
from ..Transforms import ABCTransform


def save_pil(save_imgs: List[np.ndarray], save_path: os.PathLike, metrics: dict = {}):
    if save_imgs[0].ndim == 3: # if 3D image can only save 2D
        imgs = [img[..., img.shape[-1]//2] for img in save_imgs]
    else:
        imgs = save_imgs
    
    if metrics:
        imgs = [np.ones_like(imgs[0])] + imgs
    
    txt="\n".join([f"{k}: {v:.2f}" for k,v in metrics.items()]) if metrics else None
    
    # rescale and avoid division by zero
    grid = []
    for img in imgs:
        diff = img - img.min()
        span = img.max() - img.min()
        img = np.divide(diff, span, where=span!=0)
        img = (img * 255).astype(np.uint8)
        grid.append(img)
    grid = np.concatenate(grid, axis=1)

    # edit and save image
    os.makedirs(path.dirname(save_path), exist_ok=True)
    img_obj = Image.fromarray(grid, mode='L')
    if txt is not None:
        img_draw = ImageDraw.Draw(img_obj)
        img_draw.text((2,2), txt, fill=255, align='left')  # written over left most image
    
    img_obj.save(save_path) 


def save_labeled_plt(save_imgs: List[np.ndarray], save_path: os.PathLike, labels: List[str], metrics: dict = {}, **log_kwargs):
    if save_imgs[0].ndim == 3: # if 3D image can only save 2D
        imgs = [img[..., img.shape[-1]//2] for img in save_imgs]
    else:
        imgs = save_imgs
    
    h, w = imgs[0].shape
    
    if metrics:
        imgs = [np.ones_like(imgs[0])] + imgs
        labels = ['metrics'] + labels
            
    fig = plt.figure(tight_layout=True, figsize=(2.5*len(imgs),5.5), dpi=100)

    for i, (label, img) in enumerate(zip(labels, imgs)):
        ax = fig.add_subplot(1,len(imgs),i+1)
        if label == 'metrics':
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            for m, (k, v) in enumerate(metrics.items()):
                plt.text(0,(m+1)*h//10,f"{k}: {v:.2f}", size='small', backgroundcolor='white', alpha=0.5)
        else:
            plt.imshow(img, **log_kwargs)
            # plt.colorbar()
        plt.title(label)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    fname, _ = path.splitext(path.basename(save_path))
    fig.suptitle(fname, fontsize=16)

    plt.savefig(save_path)
    plt.close(fig) 

def save_imgs(imgs: List[np.ndarray],
              save_path: os.PathLike, 
              labels: List[str]=[], 
              metrics: dict = {}, 
              method: Literal['pil','plt','cv2'] = 'pil', 
              **log_images_kwargs):
    if method == 'pil':
        save_pil(imgs, save_path, metrics)
    elif method == 'plt':
        # TEXT = "\n".join([f"{k}: {v:.2f}" for k,v in metrics.items()])
        save_labeled_plt(imgs, save_path, labels=labels, metrics=metrics, **log_images_kwargs)
    elif method == 'cv2':
        raise NotImplementedError("OpenCV image saving not implemented")
    else:
        raise ValueError(f"Method {method} not supported")

def apply_transforms(imgs: List[np.ndarray], view_transforms: List[ABCTransform]):
    out = {'raw': ([np.squeeze(img) for img in imgs], {'cmap': 'gray'})}
    for transform in view_transforms:
        transf_name = transform.__class__.__name__
        out[transf_name] = ([np.squeeze(transform(img)) for img in imgs], transform.plotting_kwargs)
    return out