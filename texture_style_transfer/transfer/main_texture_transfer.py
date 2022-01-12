
import os
import torch
import torch.nn.functional as F
import argparse
import imageio
import numpy as np
import tqdm

import neural_renderer
from transfer.model_ortho_RE import RendererModel
from transfer.losses import compute_tv_loss
from transfer.utils import *
from transfer.vgg_caffe import VGGLoss_caffe_4_multiview
from pytorch_STROTSS_improved.strotss_interface import do_strotss

            
def run(name_one_data, lr_textures=0.025, iteration=200, image_size=512, texture_size=14):

    # assign file naming formats
    path_mesh = os.path.join(dir_data, name_one_data, name_one_data + '_face_fit_ortho_deform.obj')
    path_texture = os.path.join(os.path.dirname(path_mesh), name_one_data + '_face.jpg')
    path_style = os.path.join(os.path.dirname(path_mesh), name_one_data + '_style.jpg')
    path_landmarks_style = os.path.join(os.path.dirname(path_mesh), name_one_data + '_style.txt')
    path_mesh_output = os.path.join(os.path.dirname(path_mesh), 'result_texture_transfer', name_one_data + '_face_fit_ortho_final.obj')
    dir_output = os.path.dirname(path_mesh_output)
    os.makedirs(dir_output, exist_ok=True)

    texture_image = imageio.imread(path_texture)[:,:,0:3]/255
    texture_image = texture_image.transpose(2, 0, 1)
    texture_image = np.expand_dims(texture_image, axis=0)
    texture_image = torch.from_numpy(texture_image).float().cuda()

    style_image = imageio.imread(path_style)[:,:,0:3]/255
    style_image = style_image.transpose(2, 0, 1)
    style_image = np.expand_dims(style_image, axis=0)
    style_image = torch.from_numpy(style_image).float().cuda()

    path_style_masked = os.path.join(dir_output, 'image_transfer/style_masked.png')
    path_mask_style = os.path.join(dir_output, 'image_transfer/mask_style.png')
    mask_style = erase_background(path_style, path_landmarks_style, path_style_masked, path_mask_style, save=True)

    textured_background = imageio.imread('data/textured_background.png')[:,:,0:3]/255
    textured_background = textured_background.transpose(2, 0, 1)
    textured_background = np.expand_dims(textured_background, axis=0)
    textured_background = torch.from_numpy(textured_background).float().cuda()

    azi = [180, 150, 210, 180, 180]
    ele = [0, 0, 0, 20, -20]

    renderer_model = RendererModel(
        filename_mesh=path_mesh,
        texture_image=texture_image,
        texture_size=texture_size,
        image_size=image_size,
        ).cuda()

    strotss_img_list = []
    vertices = renderer_model.vertices.unsqueeze(0)
    faces = renderer_model.faces.unsqueeze(0)
    colors = F.grid_sample(renderer_model.texture_image, renderer_model.texture_grid).permute(0, 2, 3, 1)
    textures = colors.reshape(
        (1, colors.shape[1], renderer_model.texture_size, renderer_model.texture_size,
         renderer_model.texture_size, 3))
    loop = tqdm.tqdm(range(0,azi.__len__()))
    for _, num in enumerate(loop):
        loop.set_description('Starting Image Transfer...')
        path_stylized_view = os.path.join(dir_output, 'image_transfer/stylized_view_%d.png' % num)
        os.makedirs(os.path.dirname(path_stylized_view), exist_ok=True)
        if not os.path.exists(path_stylized_view):
            renderer_model.renderer.background_color = style_image.mean(dim=(0,2,3))
            images = renderer_model.renderer.render_rgb(vertices, faces, textures,
                                                        azimuth=azi[num], elevation=ele[num], show_forehead=True)
            images = torch.flip(images, [3])
            
            renderer_model.renderer.background_color = (1, 1, 1)
            images_silhouette = renderer_model.renderer.render_rgb(vertices, faces, torch.zeros_like(textures),
                                                        azimuth=azi[num], elevation=ele[num], show_forehead=True)
            images_silhouette = images_silhouette > 0.5
            images_silhouette = images_silhouette.float()
            images_silhouette = torch.flip(images_silhouette, [3])

            noise_intensity = 0.2
            images = (images + textured_background * noise_intensity - noise_intensity / 2) * (1 - images_silhouette) + textured_background * images_silhouette

            image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
            image = np.clip(image, 0, 1) * 255
            path_defrom = os.path.join(dir_output, 'image_transfer/defromed_view_%d.png' % num)
            os.makedirs(os.path.dirname(path_defrom), exist_ok=True)
            imageio.imsave(path_defrom, image.astype(np.uint8))

            image_silhouette = images_silhouette.detach().cpu().numpy()[0].transpose((1, 2, 0))
            image_silhouette = image_silhouette * 255
            path_silhouette = os.path.join(dir_output, 'image_transfer/mask_content_%d.png' % num)
            os.makedirs(os.path.dirname(path_silhouette), exist_ok=True)
            imageio.imsave(path_silhouette, image_silhouette.astype(np.uint8))
            do_strotss(path_defrom, path_style, path_silhouette, path_mask_style, path_stylized_view, content_weight=1.2)
        transfer_img = imageio.imread(path_stylized_view)[:, :, 0:3]
        transfer_img = transfer_img.transpose(2, 0, 1) / 255
        transfer_img = np.expand_dims(transfer_img, axis=0)
        transfer_img = torch.from_numpy(transfer_img).float().cuda()
        transfer_img = F.interpolate(transfer_img, size=(image_size, image_size))
        strotss_img_list.append(transfer_img)

    optimizer = torch.optim.Adam([
        {'params': renderer_model.texture_image, 'lr': lr_textures}])

    aov_transfer_results = strotss_img_list
    vgg_model = VGGLoss_caffe_4_multiview(aov_transfer_results, layers=['r11', 'r21', 'r31', 'r41', 'r51'])
    
    mask_style = mask_style.transpose(2, 0, 1)/255
    mask_style = np.expand_dims(mask_style, axis=0)
    mask_style = torch.from_numpy(mask_style).float().cuda()    
    renderer_model.renderer.background_color = (style_image * (1 - mask_style[:,:1,:,:])).sum(dim=(0,2,3)) / (1 - mask_style[:,0,:,:]).sum()
    
    loop = tqdm.tqdm(range(iteration))
    for j in loop:
        renderer_model.texture_image.data.clamp_(0,1)
        optimizer.zero_grad()
        loss = 0
        for aov_id in range(0,azi.__len__()):
            images = renderer_model(azi=azi[aov_id], ele=ele[aov_id])
            style_transfer_loss = vgg_model(images, aov_id)
            tv_loss = 0.02 * compute_tv_loss(images)
            tv_loss_texture = 0.01 * compute_tv_loss(renderer_model.texture_image)
            loss = loss + style_transfer_loss + tv_loss + tv_loss_texture
        loop.set_description('Loss: %.4f' % loss.item())
        loss.backward()
        optimizer.step()
        
        # if j % (iteration / 20) == 0 or j == iteration-1:
        #     if images is not None:
                # images = renderer_model(azi=180, ele=0)
                # image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
                # image = np.clip(image, 0, 1) * 255
                # imageio.imsave(os.path.join(dir_output, 'mid_180_%04d.png' % j),
                #                image.astype(np.uint8))

    # draw object
    result_vertices = renderer_model.vertices.unsqueeze(0)
    result_faces = renderer_model.faces.unsqueeze(0)
    colors = F.grid_sample(renderer_model.texture_image, renderer_model.texture_grid).permute(0, 2, 3, 1)
    result_textures = colors.reshape(
        (1, colors.shape[1], renderer_model.texture_size, renderer_model.texture_size, renderer_model.texture_size, 3))
    renderer_model.renderer.background_color = (1, 1, 1)
    loop = tqdm.tqdm(range(0, azi.__len__(), 1))
    for num, azimuth in enumerate(loop):
        loop.set_description('Drawing')
        images = renderer_model.renderer.render_rgb(result_vertices, result_faces, result_textures, azimuth=azi[num], elevation=ele[num], show_forehead=True)
        images = torch.flip(images, [3])
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        image = np.clip(image, 0, 1) * 255
        imageio.imsave('%s/res_%02d.png' % (dir_output, num), image.astype(np.uint8))
        # images_notexture = renderer_model.renderer.render_rgb(result_vertices, result_faces, torch.ones_like(result_textures), azimuth=azi[num], elevation=ele[num], lighting=True, show_forehead=True)
        # images_notexture = torch.flip(images_notexture, [3])
        # images_notexture = images_notexture.detach().cpu().numpy()[0].transpose((1, 2, 0))
        # images_notexture = np.clip(images_notexture, 0, 1) * 255
        # imageio.imsave('%s/res_not_%02d.png' % (dir_output, num), images_notexture.astype(np.uint8))
        # 
        # if num == 0:
        #     images = renderer_model.renderer.render_rgb(result_vertices, result_faces, result_textures, azimuth=azi[num], elevation=ele[num], show_forehead=False)
        #     images = torch.flip(images, [3])
        #     image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        #     image = np.clip(image, 0, 1) * 255
        #     imageio.imsave('%s/res_aligned_%02d.png' % (dir_output, num), image.astype(np.uint8))
        # 
        #     images_silhouette = renderer_model.renderer.render_rgb(result_vertices, result_faces, torch.zeros_like(result_textures), azimuth=azi[num], elevation=ele[num], show_forehead=False)
        #     images_silhouette = images_silhouette > 0.5
        #     images_silhouette = images_silhouette.float()
        #     images_silhouette = torch.flip(images_silhouette, [3])
        #     image_silhouette = images_silhouette.detach().cpu().numpy()[0].transpose((1, 2, 0))
        #     image_silhouette = image_silhouette * 255
        #     imageio.imsave('%s/res_aligned_mask_%02d.png' % (dir_output, num), image_silhouette.astype(np.uint8))

    neural_renderer.save_obj(path_mesh_output, result_vertices[0]/800*512, result_faces[0],
                             torch.clamp(result_textures[0], 0, 1) * 255)
    texture_image = renderer_model.texture_image.detach().cpu().numpy()[0].transpose((1, 2, 0))
    texture_image = np.clip(texture_image, 0, 1) * 255
    imageio.imsave('%s/texture_image.png' % dir_output, texture_image.astype(np.uint8))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--dir_data', type=str, default='../data_demo/')
    args = parser.parse_args()

    dir_data = args.dir_data
    dirs_data_sub = os.listdir(dir_data)
    dirs_data_sub.sort()
    for dir_data_sub in dirs_data_sub:
        if dir_data_sub[0] != '.':
            print('-----Now Transfer: ', dir_data_sub, '-----')
            run(dir_data_sub)
