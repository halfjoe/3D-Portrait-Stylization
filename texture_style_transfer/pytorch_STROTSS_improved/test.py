import argparse
import imageio
import numpy as np
import torch

import style_transfer
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser('Style transfer by relaxed optimal transport with sinkhorn distance')
    parser.add_argument('--content', '-c', help="path of content img", required=True)
    parser.add_argument('--style', '-s', help="path of style img", required=True)
    parser.add_argument('--output', '-o', help="path of output img", default='output.png')
    parser.add_argument('--content_weight', type=float, help='no padding used', default=0.5)
    parser.add_argument('--max_scale', type=int, help='max scale for the style transfer', default=4)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--content_guidance', default='', help="path of content guidance region image")
    parser.add_argument('--style_guidance', default='', help="path of style guidance regions image")
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency for the loss')
    parser.add_argument('--use_sinkhorn', action='store_true', help='use sinkhorn algo. for the earth mover distance')
    parser.add_argument('--sinkhorn_reg', type=float, help='reg param for sinkhorn', default=0.1)
    parser.add_argument('--sinkhorn_maxiter', type=int, default=30, help='number of interations for sinkohrn algo')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    content_weight = 16 * args.content_weight
    max_scale = args.max_scale
    use_guidance_region = args.content_guidance and args.style_guidance

    if use_guidance_region:
        content_regions, style_regions = utils.extract_regions(args.content_guidance, args.style_guidance)
    else:
        content_img, style_img = imageio.imread(args.content), imageio.imread(args.style)
        content_regions, style_regions = [np.ones(content_img.shape[:2], dtype=np.float32)], [np.ones(style_img.shape[:2], dtype=np.float32)]

    loss, canvas = style_transfer.run_style_transfer(args.content, args.style, content_weight,
            max_scale, content_regions, style_regions, args.output,
            print_freq=args.print_freq, use_sinkhorn=args.use_sinkhorn,
            sinkhorn_reg=args.sinkhorn_reg, sinkhorn_maxiter=args.sinkhorn_maxiter
        )
