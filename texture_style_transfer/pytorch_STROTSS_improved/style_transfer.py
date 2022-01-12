import glob
import imageio
import numpy as np
import time
import torch
import torch.nn.functional as F

from pytorch_STROTSS_improved import utils
from pytorch_STROTSS_improved import vgg_pt
from pytorch_STROTSS_improved import loss_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def style_transfer(stylized_im, content_im, style_path, output_path,
                   long_side, content_weight, content_regions, style_regions,
                   lr, print_freq=100, max_iter=250,
                   resample_freq=1, optimize_laplacian_pyramid=True,
                   use_sinkhorn=False, sinkhorn_reg=0.1, sinkhorn_maxiter=30):

    cnn = vgg_pt.Vgg16_pt().to(device)

    phi = lambda x: cnn.forward(x)
    phi2 = lambda x, y, z: cnn.forward_cat(x, z, samps=y, forward_func=cnn.forward)


    if optimize_laplacian_pyramid:
        laplacian_pyramid = utils.create_laplacian_pyramid(stylized_im, pyramid_depth=5)
        parameters = [torch.nn.Parameter(li.data, requires_grad=True) for li in laplacian_pyramid]
    else:
        parameters = [torch.nn.Parameter(stylized_im.data, requires_grad=True)]

    optimizer = torch.optim.RMSprop(parameters, lr=lr)

    content_im_cnn_features = cnn(content_im)

    style_image_paths = glob.glob(style_path+'*')[::3]

    strotss_loss = loss_utils.RelaxedOptimalTransportSelfSimilarityLoss(
            use_sinkhorn=use_sinkhorn, sinkhorn_reg=sinkhorn_reg, sinkhorn_maxiter=sinkhorn_maxiter)

    style_features = []
    for style_region in style_regions:
        style_features.append(utils.load_style_features(phi2, style_image_paths, style_region,
                                                        subsamps=1000, scale=long_side, inner=5))

    if optimize_laplacian_pyramid:
        stylized_im = utils.synthetize_image_from_laplacian_pyramid(parameters)
    else:
        stylized_im = parameters[0]

    resized_content_regions = []
    for content_region in content_regions:
        resized_content_region = utils.resize(torch.from_numpy(content_region), (stylized_im.size(3), stylized_im.size(2)), mode='nearest').numpy()
        resized_content_regions.append(resized_content_region.astype('bool'))

    for i in range(max_iter):
        if i == 200:
            optimizer = torch.optim.RMSprop(parameters, lr=0.1*lr)

        optimizer.zero_grad()
        if optimize_laplacian_pyramid:
            stylized_im = utils.synthetize_image_from_laplacian_pyramid(parameters)
        else:
            stylized_im = parameters[0]

        if i == 0 or i % (resample_freq*10) == 0:
            for i_region, resized_content_region in enumerate(resized_content_regions):
                strotss_loss.init_inds(content_im_cnn_features, style_features[i_region], resized_content_region, i_region)

        if i == 0 or i % resample_freq == 0:
            strotss_loss.shuffle_feature_inds()

        stylized_im_cnn_features = cnn(stylized_im)

        loss = strotss_loss.eval(stylized_im_cnn_features,
                content_im_cnn_features, style_features,
                content_weight=content_weight, moment_weight=1.0)

        loss.backward()
        optimizer.step()


        if i % print_freq == 0:
            print(f'step {i}/{max_iter}, loss {loss.item():.6f}')

    return stylized_im, loss


def run_style_transfer(content_path, style_path, content_weight, max_scale, content_regions, style_regions, output_path='./output.png', print_freq=100, use_sinkhorn=False, sinkhorn_reg=0.1, sinkhorn_maxiter=30):

    smallest_size = 64
    start = time.time()

    content_image, style_image = utils.load_img(content_path), utils.load_img(style_path)
    _, content_H, content_W = content_image.size()
    _, style_H, style_W = style_image.size()
    print(f'content image size {content_H}x{content_W}, style image size {style_H}x{style_W}')

    for scale in range(1, max_scale+1):
        t0 = time.time()

        scaled_size = smallest_size*(2**(scale-1))

        print('Processing scale {}/{}, size {}...'.format(scale, max_scale, scaled_size))

        content_scaled_size = (int(content_H * scaled_size / content_W), scaled_size) if content_H < content_W else (scaled_size , int(content_W * scaled_size / content_H))
        content_image_scaled = utils.resize(content_image.unsqueeze(0), content_scaled_size).to(device)
        bottom_laplacian = content_image_scaled - utils.resize(utils.downsample(content_image_scaled), content_scaled_size)

        lr = 2e-3
        if scale == 1:
            style_image_mean = style_image.unsqueeze(0).mean(dim=(2, 3), keepdim=True).to(device)
            stylized_im = style_image_mean + bottom_laplacian
        elif scale > 1 and scale < max_scale:
            stylized_im = utils.resize(stylized_im.clone(), content_scaled_size) + bottom_laplacian
        elif scale == max_scale:
            stylized_im = utils.resize(stylized_im.clone(), content_scaled_size)
            lr = 1e-3

        stylized_im, final_loss = style_transfer(stylized_im, content_image_scaled, style_path, output_path, scaled_size, content_weight, content_regions, style_regions, lr, print_freq=print_freq, use_sinkhorn=use_sinkhorn, sinkhorn_reg=sinkhorn_reg, sinkhorn_maxiter=sinkhorn_maxiter)

        content_weight /= 2.0
        print('...done in {:.1f} sec, final loss {:.4f}'.format(time.time()-t0, final_loss.item()))

    print('Finished in {:.1f} secs' .format(time.time()-start))

    canvas = torch.clamp(stylized_im[0],-0.5,0.5).data.cpu().numpy().transpose(1,2,0)
    print(f'Saving to output to {output_path}.')
    imageio.imwrite(output_path, canvas)

    return final_loss, stylized_im
