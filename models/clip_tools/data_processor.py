import os
import PIL
import math
import torch
import random
import jsonlines
from tqdm import tqdm
from einops import rearrange
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import CLIPImageProcessor

random.seed(1998)
def crop_img(img):
    inv_img = ImageOps.invert(img)
    bbox = inv_img.getbbox()
    bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)
    img = inv_img.crop(bbox)
    img = ImageOps.invert(img)
    return img


def process_figure(processor: CLIPImageProcessor, figure_path, aug=False, crop=False, white=False, output=False):
    img = Image.open(figure_path)
    if crop:
        img = crop_img(img)
    if white:
        tmp_img = Image.new('RGB', img.size, (255, 255, 255))
        tmp_img.paste(img, (0, 0), mask=img)
        img = tmp_img

    if aug:
        img = img.rotate(random.randint(0, 3) * 90, expand=True)
        if random.random() <= 0.5:
            img = img.convert('L').convert('RGB')
        img = transforms.ColorJitter(brightness=[0.75, 1.25], contrast=0, saturation=0, hue=0)(img)

    width, height = img.size
    if width < height:
        new_width, new_height = width, math.ceil(height / width) * width
    else:
        new_width, new_height = math.ceil(width / height) * height, height
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    new_img.paste(img, (0, 0))

    if output:
        new_img.save(f'./temp/{"_".join(figure_path.split("/")[-2:])}')

    result = processor(new_img, do_center_crop=False, return_tensors='pt')['pixel_values']
    _, _, h, w = result.shape
    assert h == 336 or w == 336
    assert h % 336 ==0 and w % 336 == 0
    if h == 336 and w == 336:
        return result, 1

    square = max(width, height)
    new_img = Image.new('RGB', (square, square), (255, 255, 255))
    new_img.paste(img, (0, 0))
    full_figure = processor(new_img, do_center_crop=False, return_tensors='pt')['pixel_values']
    assert full_figure.shape[2] == 336 and full_figure.shape[3] == 336
    if h == 336:
        assert height < width
        sub_figures = rearrange(result, 'b c h (w n_w) -> (b n_w) c h w', w=336)
    else:
        assert width < height and w == 336
        sub_figures = rearrange(result, 'b c (h n_h) w -> (b n_h) c h w', h=336)
    returns = torch.cat([full_figure, sub_figures], dim=0)
    dims = returns.shape
    assert dims[2] == 336 and dims[3] == 336 and dims[1] == 3 and len(dims) == 4, dims
    return returns, returns.shape[0] if width > height else -returns.shape[0]
