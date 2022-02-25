import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from vgg_caffe import VGGLoss_caffe

VGG_MEAN = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], requires_grad=False).cuda()
VGG_STD = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], requires_grad=False).cuda()
# VGG_STD = torch.tensor([[[1.]], [[1.]], [[1.]]], requires_grad=False).cuda()


class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            if laplacian[i, i] == 0:
                laplacian[i, :] /= 1e-4
            else:
                laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian).cuda())

    def forward(self, x, targ):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        targ = torch.matmul(self.laplacian, targ)
        dims = tuple(range(x.ndimension())[1:])
        diff = x - targ
        diff = diff.pow(2).sum(dims)
        if self.average:
            return diff.sum() / batch_size
        else:
            return diff


class StyleLossGatys(nn.Module):
    def __init__(self, style_image):
        super(StyleLossGatys, self).__init__()
        self.VGG = VGGLoss_caffe().cuda()
        style_features_target = self.VGG.get_style_features(style_image)
        self.gram_matrix_target = [gram_matrix(y).detach() for y in style_features_target]
        self.mse_loss = nn.MSELoss()

    def forward(self, image):
        style_features = self.VGG.get_style_features(image)
        style_loss = 0
        i = 0
        w = [64, 128, 256, 512, 512]
        for ft_y, gm_s in zip(style_features, self.gram_matrix_target):
            gm_y = gram_matrix(ft_y)
            style_loss += 1e3/w[i]**2 * self.mse_loss(gm_y, gm_s)
            i += 1
        return style_loss


def gram_matrix(y):
    b, c, h, w = y.size()
    F = y.view(b, c, h * w)
    G = torch.bmm(F, F.transpose(1, 2))
    G.div_(h * w)
    return G


class ContentLossGatys(nn.Module):
    def __init__(self, content_image, face_mask=None):
        super(ContentLossGatys, self).__init__()
        self.VGG19 = VGGLoss_caffe().cuda()
        self.content_features_target = self.VGG19.get_content_feature(content_image)
        self.mse_loss = nn.MSELoss()
        self.face_mask = face_mask

    def forward(self, image, face_with_background=None, pad_face=False):
        content_feature = self.VGG19.get_content_feature(image)
        if pad_face:
            face_with_background = face_with_background.sub(VGG_MEAN).div(VGG_STD)
            face_content_features = self.VGG19.get_content_features(face_with_background)
            face_feature_mask = F.interpolate(self.face_mask, size=(face_content_features.shape[2], face_content_features.shape[3]))
            content_feature = (1-face_feature_mask)*content_feature+face_feature_mask*face_content_features
        content_loss = self.mse_loss(content_feature[0], self.content_features_target[0])
        return content_loss


# def compute_tv_loss(images, masks):
#     s1 = torch.pow(images[:, :, 1:, :-1] - images[:, :, :-1, :-1], 2)
#     s2 = torch.pow(images[:, :, :-1, 1:] - images[:, :, :-1, :-1], 2)
#     masks = masks[:, None, :-1, :-1].repeat(1, s1.shape[1], 1, 1)
#     return torch.sum(masks * (s1 + s2))

def compute_tv_loss(images):
    s1 = torch.pow(images[:, :, 1:, :-1] - images[:, :, :-1, :-1], 2)
    s2 = torch.pow(images[:, :, :-1, 1:] - images[:, :, :-1, :-1], 2)
    return torch.sum(s1 + s2)
