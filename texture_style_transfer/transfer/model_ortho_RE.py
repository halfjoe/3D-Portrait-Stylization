
import random
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

import neural_renderer
from utils import *


class RendererModel(nn.Module):
    def __init__(
            self,
            filename_mesh,
            texture_image,
            texture_size=4,
            image_size=512,
            batch_size=1,
            ):
        super(RendererModel, self).__init__()

        self.texture_size = texture_size
        self.image_size = image_size
        self.batch_size = batch_size

        # load .obj
        mesh = neural_renderer.Mesh.fromobj(filename_mesh, load_texture=True, normalization=False, texture_size=texture_size)
        self.register_parameter('vertices', nn.Parameter(mesh.vertices))
        original_vertices = torch.empty_like(mesh.vertices).copy_(mesh.vertices)

        self.register_buffer('original_vertices', original_vertices)
        self.register_buffer('faces', mesh.faces)

        texture_grid = mesh.textures[:, :, :, :, 0:2] / (texture_image.shape[-1] / 2) - 1
        texture_grid[:, :, :, :, 1] = 0 - texture_grid[:, :, :, :, 1]
        texture_grid = texture_grid.reshape((texture_grid.shape[0], -1, texture_grid.shape[-1])).unsqueeze(0)
        self.register_buffer('texture_grid', texture_grid)
        self.register_parameter('texture_image', nn.Parameter(texture_image))

        # setup renderer
        renderer = neural_renderer.Renderer(background_color=[0,0,0],
                                            image_size=self.image_size,
                                            is_ortho=True)
        self.renderer = renderer
        self.elevation = 0
        self.azimuth = 180

    def forward(self, azi=180, ele=0, show_forehead=False):

        self.azimuth = azi
        self.elevation = ele

        # get images
        colors = F.grid_sample(self.texture_image, self.texture_grid).permute(0, 2, 3, 1)
        colors = colors.reshape((1, colors.shape[1], self.texture_size, self.texture_size, self.texture_size, 3))
        strangerepeatingtextures = colors.repeat(self.batch_size, 1, 1, 1, 1, 1)
        images = self.renderer.render_rgb(self.vertices.repeat(self.batch_size, 1, 1),
                                          self.faces.repeat(self.batch_size, 1, 1), strangerepeatingtextures,
                                          elevation=self.elevation, azimuth=self.azimuth, show_forehead=True)
        images = torch.flip(images, [3])
        return images
