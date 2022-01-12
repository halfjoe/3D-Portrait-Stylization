
import torch
import torch.nn.functional as F

from collections import namedtuple

def vgg_preprocess_caffe(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat(
        (tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    tensor_bgr_ml = tensor_bgr - \
        torch.Tensor([0.40760392, 0.45795686, 0.48501961]
                     ).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst

class VGG19_caffe(torch.nn.Module):
    '''
    NOTE: input tensor should range in [0,1]
    '''

    def __init__(self, pool='max'):
        super(VGG19_caffe, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        '''
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess_caffe(x)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class VGGLoss_caffe(torch.nn.Module):
    def __init__(self):
        super(VGGLoss_caffe, self).__init__()
        self.vgg = VGG19_caffe()
        self.vgg.load_state_dict(torch.load('models/vgg19_conv.pth'))
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        features_A = self.vgg(x, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True)
        features_B = self.vgg(y, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True)
        loss = 0
        for i in range(len(features_A)):
            loss += self.weights[i] * self.criterion(features_A[i], features_B[i].detach())
        return loss

    def get_style_features(self, x, layers=['r11', 'r21', 'r31', 'r41', 'r51']):
        return self.vgg(x, layers, preprocess=True)

    def get_content_feature(self, x, layers=['r42']):
        return self.vgg(x, layers, preprocess=True)


class VGGLoss_caffe_4_multiview(torch.nn.Module):
    def __init__(self, multiview_style_images, layers=['r21', 'r31', 'r41', 'r51']):
        super(VGGLoss_caffe_4_multiview, self).__init__()
        self.vgg = VGG19_caffe()
        self.vgg.load_state_dict(torch.load('transfer/models/vgg19_conv.pth'))
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.cuda()
        self.criterion = torch.nn.L1Loss()
        self.layers = layers
        self.weights = [5.0, 5.0, 1.0, 1.0, 1.0]
        multiview_style_features = []
        for style_image in multiview_style_images:
            style_feature = self.vgg(style_image, layers, preprocess=True)
            multiview_style_features.append(style_feature)
        self.multiview_style_features = multiview_style_features

    def forward(self, x, aov_id):
        features = self.vgg(x, self.layers, preprocess=True)
        loss = 0
        for i in range(len(features)):
            loss += self.weights[i] * self.criterion(features[i], self.multiview_style_features[aov_id][i].detach())
        #     print(loss.item())
        return loss
