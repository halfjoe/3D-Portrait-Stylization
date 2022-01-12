
import argparse
import imageio
import numpy as np
import torch

from pytorch_STROTSS_improved import style_transfer
from pytorch_STROTSS_improved import utils

def do_strotss(content, style, content_guidance='', style_guidance='', output='output.png', content_weight=1.2):

    seed = 0
    content_weight = content_weight
    max_scale = 4
    print_freq = 100
    use_sinkhorn = False
    sinkhorn_reg = 0.1
    sinkhorn_maxiter = 30
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    content_weight = 16 * content_weight
    max_scale = max_scale
    use_guidance_region = content_guidance and style_guidance

    if use_guidance_region:
        content_regions, style_regions = utils.extract_regions(content_guidance, style_guidance)
    else:
        content_img, style_img = imageio.imread(content), imageio.imread(style)
        content_regions, style_regions = [np.ones(content_img.shape[:2], dtype=np.float32)], [np.ones(style_img.shape[:2], dtype=np.float32)]

    loss, canvas = style_transfer.run_style_transfer(content, style, content_weight,
            max_scale, content_regions, style_regions, output,
            print_freq=print_freq, use_sinkhorn=use_sinkhorn,
            sinkhorn_reg=sinkhorn_reg, sinkhorn_maxiter=sinkhorn_maxiter
        )
