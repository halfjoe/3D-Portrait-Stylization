import imageio
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def downsample(img, factor=2, mode='bilinear'):
    img_H, img_W = img.size(2), img.size(3)
    return F.interpolate(img, (max(img_H//factor, 1), max(img_W//factor, 1)), mode=mode)


def resize(img, size, mode='bilinear'):
    if len(img.shape) == 2:
        return F.interpolate(img.unsqueeze(0).unsqueeze(0), size, mode=mode)[0, 0]
    elif len(img.shape) == 3:
        return F.interpolate(img.unsqueeze(0), size, mode=mode)[0]
    return F.interpolate(img, size, mode=mode)


def load_img(img_path, size=None):
    img = torchvision.transforms.functional.to_tensor(Image.open(img_path).convert('RGB')) - 0.5
    if size is None:
        return img
    elif isinstance(size, (int, float)):
        return F.interpolate(img.unsqueeze(0), scale_factor=size/img.size(1), mode='bilinear')[0]
    else:
        return F.interpolate(img.unsqueeze(0), size, mode='bilinear')[0]


def create_laplacian_pyramid(image, pyramid_depth):
    laplacian_pyramid = []
    current_image = image
    for i in range(pyramid_depth):
        laplacian_pyramid.append(current_image - resize(downsample(current_image), current_image.shape[2:4]))
        current_image = downsample(current_image)
    laplacian_pyramid.append(current_image)

    return laplacian_pyramid


def synthetize_image_from_laplacian_pyramid(laplacian_pyramid):
    current_image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid)-2, -1, -1):
        up_x = laplacian_pyramid[i].size(2)
        up_y = laplacian_pyramid[i].size(3)
        current_image = laplacian_pyramid[i] + resize(current_image, (up_x,up_y))

    return current_image

YUV_transform = torch.from_numpy(np.float32([
    [0.577350,0.577350,0.577350],
    [-0.577350,0.788675,-0.211325],
    [-0.577350,-0.211325,0.788675]
])).to(device)


def rgb_to_yuv(rgb):
    global YUV_transform
    return torch.mm(YUV_transform, rgb)


def extract_regions(content_path, style_path, min_count=10000):
    style_guidance_img = imageio.imread(style_path).transpose(1,0,2)
    content_guidance_img = imageio.imread(content_path).transpose(1,0,2)

    color_codes, color_counts = np.unique(style_guidance_img.reshape(-1, style_guidance_img.shape[2]), axis=0, return_counts=True)

    color_codes = color_codes[color_counts > min_count]

    content_regions = []
    style_regions = []

    for color_code in color_codes:
        color_code = color_code[np.newaxis, np.newaxis, :]

        style_regions.append((np.abs(style_guidance_img - color_code).sum(axis=2) == 0).astype(np.float32))
        content_regions.append((np.abs(content_guidance_img - color_code).sum(axis=2) == 0).astype(np.float32))

    return [content_regions, style_regions]


def load_style_features(features_extractor, paths, style_region, subsamps=-1, scale=-1, inner=1):
    features = []

    for p in paths:
        style_im = load_img(p, size=scale).unsqueeze(0).to(device)

        r = resize(torch.from_numpy(style_region), (style_im.size(3), style_im.size(2))).numpy()

        # NOTE: understand inner
        for j in range(inner):
            with torch.no_grad():
                features_j = features_extractor(style_im, subsamps, r)

            features_j = [feat_j.view(feat_j.size(0), feat_j.size(1), -1, 1) for feat_j in features_j]

            if len(features) == 0:
                features = features_j
            else:
                features = [torch.cat([features_j[i], features[i]],2) for i in range(len(features))]

    return features
