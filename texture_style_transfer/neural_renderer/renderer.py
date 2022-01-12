from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr


class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0,0,0],
                 fill_back=True, camera_mode='look_at',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0,0,1],
                 near=-100, far=1000,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1,1,1], light_color_directional=[1,1,1],
                 light_direction=[0,0,1], use_facewarehouse=False, proj_param=None, trans_param=None, vertices_4_z=None,
                 landmarks_content=None, vertices_to_align=None, is_ortho=False):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back
        self.use_facewarehouse = use_facewarehouse
        self.is_ortho = is_ortho
        self.default_rotation_mat = self.get_rotation_mat(0, 180)
        if use_facewarehouse:
            self.proj_mat, self.trans_mat = self.compute_proj_trans_mat(proj_param, trans_param)
            if landmarks_content is not None:
                self.fake_proj_mat = self.get_proj_mat_to_align_content_landmark(landmarks_content, vertices_to_align, vertices_4_z)
            else:
                self.fake_proj_mat = self.get_proj_mat_fake(vertices_4_z)

        # camera
        self.camera_mode = camera_mode
        if self.camera_mode == 'projection':
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = camera_direction
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')


        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction 

        # rasterization
        self.rasterizer_eps = 1e-3
        
        self.fix_width_left = 0
        self.fix_width_right = 0

    def forward(self, vertices, faces, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''
        
        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None, elevation=0, azimuth=180, show_forehead=False):
        
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        if True:
            vertices = vertices
            # vertices = vertices/800*2
            # z = vertices[:, :, 2] - (vertices[:, :, 2].min()+vertices[:, :, 2].max())/2
            # y = vertices[:, :, 1] - 1
            # x = vertices[:, :, 0] - 1

            # vertices = torch.stack((x, y, z), dim=2)
            # vertices = self.rotate_ortho(vertices, elevation, azimuth, show_forehead)
        elif self.use_facewarehouse:
            vertices = self.transform_by_facewarehouse_param_and_rotate(vertices, elevation, azimuth, show_forehead)
        else:
            # viewpoint transformation
            if self.camera_mode == 'look_at':
                vertices = nr.look_at(vertices, self.eye)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'look':
                vertices = nr.look(vertices, self.eye, self.camera_direction)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'projection':
                if K is None:
                    K = self.K
                if R is None:
                    R = self.R
                if t is None:
                    t = self.t
                if dist_coeffs is None:
                    dist_coeffs = self.dist_coeffs
                if orig_size is None:
                    orig_size = self.orig_size
                vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None,
                   return_vertices=False, lighting=False, elevation=0, azimuth=180, show_forehead=False,
                   lighting_params=None, no_ambient=False, fix_width=False):
        if return_vertices:
            pass
        elif self.fill_back:
            # fill back
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # # lighting
        if lighting:
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            if lighting_params is not None:
                self.light_intensity_ambient = lighting_params[0]
                self.light_intensity_directional = lighting_params[1]
                self.light_direction = lighting_params[-3:]
            if no_ambient:
                textures = nr.lighting(
                    faces_lighting,
                    textures,
                    0,
                    self.light_intensity_directional,
                    self.light_color_ambient,
                    self.light_color_directional,
                    self.light_direction)
            else:
                textures = nr.lighting(
                    faces_lighting,
                    textures,
                    self.light_intensity_ambient,
                    self.light_intensity_directional,
                    self.light_color_ambient,
                    self.light_color_directional,
                    self.light_direction)

        if self.is_ortho:
            vertices = vertices/800*2
            z = vertices[:, :, 2] - (vertices[:, :, 2].min()+vertices[:, :, 2].max())/2
            y = vertices[:, :, 1] - 1
            x = vertices[:, :, 0] - 1

            vertices = torch.stack((x, y, z), dim=2)
            vertices = self.rotate_ortho(vertices, elevation, azimuth, show_forehead, fix_width)

            if return_vertices:
                return vertices
            else:
                # rasterization
                faces = nr.vertices_to_faces(vertices, faces)
                images = nr.rasterize(
                    faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
                    self.background_color)
                return images

        if self.use_facewarehouse:
            # vertices = self.transform_by_facewarehouse_param(vertices)
            # vertices = self.rotate_along_x_y(vertices, elevation, azimuth)
            vertices = self.transform_by_facewarehouse_param_and_rotate(vertices, elevation, azimuth, show_forehead)
        else:
            # viewpoint transformation
            if self.camera_mode == 'look_at':
                vertices = nr.look_at(vertices, self.eye)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'look':
                vertices = nr.look(vertices, self.eye, self.camera_direction)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'projection':
                if K is None:
                    K = self.K
                if R is None:
                    R = self.R
                if t is None:
                    t = self.t
                if dist_coeffs is None:
                    dist_coeffs = self.dist_coeffs
                if orig_size is None:
                    orig_size = self.orig_size
                vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)


        if return_vertices:
            return vertices
        else:
            # rasterization
            faces = nr.vertices_to_faces(vertices, faces)
            images = nr.rasterize(
                faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
                self.background_color)
            return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None,
               return_vertices=False, lighting=False, elevation=0, azimuth=180):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # # lighting
        if lighting:
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            textures = nr.lighting(
                faces_lighting,
                textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)

        if self.use_facewarehouse:
            # vertices = self.transform_by_facewarehouse_param(vertices)
            # vertices = self.rotate_along_x_y(vertices, elevation, azimuth)
            vertices = self.transform_by_facewarehouse_param_and_rotate(vertices, elevation, azimuth)

        else:
            # viewpoint transformation
            if self.camera_mode == 'look_at':
                vertices = nr.look_at(vertices, self.eye)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'look':
                vertices = nr.look(vertices, self.eye, self.camera_direction)
                # perspective transformation
                if self.perspective:
                    vertices = nr.perspective(vertices, angle=self.viewing_angle)
            elif self.camera_mode == 'projection':
                if K is None:
                    K = self.K
                if R is None:
                    R = self.R
                if t is None:
                    t = self.t
                if dist_coeffs is None:
                    dist_coeffs = self.dist_coeffs
                if orig_size is None:
                    orig_size = self.orig_size
                vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)


        if return_vertices:
            return vertices
        else:
            # rasterization
            faces = nr.vertices_to_faces(vertices, faces)
            out = nr.rasterize_rgbad(
                faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
                self.background_color, return_depth=False)
            return out['rgb'], out['alpha']

    def compute_proj_trans_mat(self, proj_param=None, trans_param=None):

        m_focalLen = proj_param[0]
        m_center = proj_param[1:]
        proj_mat = numpy.zeros((3,3))
        proj_mat[0,0] = m_focalLen
        proj_mat[1,1] = m_focalLen
        proj_mat[0,2] = m_center[0]
        proj_mat[1,2] = m_center[1]
        proj_mat[2,2] = 1.
        proj_mat = torch.from_numpy(proj_mat).cuda().float()

        m_r = trans_param[0]
        m_v = trans_param[1:4]
        m_trans = trans_param[4:7]
        m_scale = trans_param[-1]

        rr = m_r * m_r;
        v00 = m_v[0] * m_v[0];
        v11 = m_v[1] * m_v[1];
        v22 = m_v[2] * m_v[2];
        v01 = m_v[0] * m_v[1];
        v12 = m_v[1] * m_v[2];
        v20 = m_v[2] * m_v[0];
        rv0 = m_r * m_v[0];
        rv1 = m_r * m_v[1];
        rv2 = m_r * m_v[2];

        rotMat = numpy.zeros((9,1))
        rotMat[0] = rr + v00 - v11 - v22;
        rotMat[1] = 2. * (v01 - rv2);
        rotMat[2] = 2. * (rv1 + v20);
        rotMat[3] = 2. * (v01 + rv2);
        rotMat[4] = rr - v00 + v11 - v22;
        rotMat[5] = 2. * (v12 - rv0);
        rotMat[6] = 2. * (v20 - rv1);
        rotMat[7] = 2. * (rv0 + v12);
        rotMat[8] = rr - v00 - v11 + v22;

        trans_mat = numpy.zeros((12,1))
        trans_mat[0] = rotMat[0] * m_scale;
        trans_mat[1] = rotMat[1] * m_scale;
        trans_mat[2] = rotMat[2] * m_scale;
        trans_mat[3] = m_trans[0];
        trans_mat[4] = rotMat[3] * m_scale;
        trans_mat[5] = rotMat[4] * m_scale;
        trans_mat[6] = rotMat[5] * m_scale;
        trans_mat[7] = m_trans[1];
        trans_mat[8] = rotMat[6] * m_scale;
        trans_mat[9] = rotMat[7] * m_scale;
        trans_mat[10] = rotMat[8] * m_scale;
        trans_mat[11] = m_trans[2];

        trans_mat = trans_mat.reshape((3,4))
        trans_mat = torch.from_numpy(trans_mat).cuda().float()

        return proj_mat, trans_mat

    def transform_by_facewarehouse_param(self, vertices):

        # proj_trans_mat = torch.matmul(proj_mat, trans_mat)
        vertices = torch.cat((vertices, torch.ones_like(vertices[:,:,1]).unsqueeze(2)),dim=2)
        vertices = torch.matmul(self.trans_mat, vertices.permute((0,2,1))).permute(0,2,1)
        vertices = vertices * 0.9
        vertices[:,:,1] = vertices[:,:,1] - 0.1

        # vertices = torch.matmul(proj_mat, vertices.permute((0,2,1))).permute(0,2,1)

        # z = vertices[:, :, 2]
        # vertices = torch.cat((vertices[:,:,0:2], torch.ones_like(vertices[:,:,1]).unsqueeze(2)),dim=2)
        # vertices = torch.matmul(self.proj_mat, vertices.permute((0,2,1))).permute(0,2,1)/1600
        # vertices = torch.stack((vertices[:,:,0], vertices[:,:,1], z), dim=2)

        return vertices

    def get_rotation_mat(self, elevation, azimuth):

        elevation = math.radians(elevation)
        azimuth = math.radians(azimuth)

        rotate_mat_x = numpy.zeros((3,3))
        rotate_mat_x[0,0] = 1
        rotate_mat_x[1,1] = math.cos(elevation)
        rotate_mat_x[1,2] = -math.sin(elevation)
        rotate_mat_x[2,1] = math.sin(elevation)
        rotate_mat_x[2,2] = math.cos(elevation)
        rotate_mat_x = torch.from_numpy(rotate_mat_x).cuda()

        rotate_mat_y = numpy.zeros((3,3))
        rotate_mat_y[0,0] = math.cos(azimuth)
        rotate_mat_y[0,2] = math.sin(azimuth)
        rotate_mat_y[1,1] = 1
        rotate_mat_y[2,0] = -math.sin(azimuth)
        rotate_mat_y[2,2] = math.cos(azimuth)
        rotate_mat_y = torch.from_numpy(rotate_mat_y).cuda()

        rotate_mat = rotate_mat_x.mm(rotate_mat_y).float()

        return rotate_mat

    def get_proj_mat_fake(self, vertices_4_z):
        # cheat projection mat
        proj_mat_fake = torch.zeros((3,4)).cuda().float()
        proj_mat_fake[0,0] = 0.9
        proj_mat_fake[1,1] = 0.9
        proj_mat_fake[2,2] = 0.9
        proj_mat_fake[1, 3] = - 0.1
        if vertices_4_z is not None:
            vertices_4_z = self.transform_by_facewarehouse_param(vertices_4_z.unsqueeze(0))
            proj_mat_fake[2, 3] = - vertices_4_z[:,:,2].mean()
        else:
            proj_mat_fake[2, 3] = 32
        return proj_mat_fake

    def get_proj_mat_to_align_content_landmark(self, target_landmarks, vertices_to_align, vertices_4_z):

        target_landmarks = target_landmarks * -2 + 1
        vertices_to_align = vertices_to_align.unsqueeze(0)

        vertices = torch.cat((vertices_to_align, torch.ones_like(vertices_to_align[:,:,1]).unsqueeze(2)),dim=2)
        trans_mat = torch.cat((self.trans_mat, torch.from_numpy(numpy.array([0,0,0,1])).cuda().float().unsqueeze(0)), dim=0)
        rot_mat = torch.cat((self.default_rotation_mat, torch.from_numpy(numpy.array([0,0,1])).cuda().float().unsqueeze(0).t()), dim=1)
        total_mat = rot_mat.mm(trans_mat)
        vertices = torch.matmul(total_mat, vertices.permute((0,2,1))).permute(0,2,1)
        source_landmarks = vertices.squeeze(0)[:,:2]
        # source_landmarks[:,0] = - source_landmarks[:,0]
        # source_landmarks = source_landmarks * -1/2 + 1/2

        # shrink
        shrink_ratio = (target_landmarks[1, 1] - target_landmarks[0, 1]) / (
                    source_landmarks[1, 1] - source_landmarks[0, 1])
        x_translation = target_landmarks[0, 0] - source_landmarks[0,0] * shrink_ratio
        y_translation = target_landmarks[0, 1] - source_landmarks[0,1] * shrink_ratio

        # projection mat
        proj_mat = torch.zeros((3,4)).cuda().float()
        proj_mat[0,0] = shrink_ratio
        proj_mat[1,1] = shrink_ratio
        proj_mat[2,2] = shrink_ratio
        proj_mat[2,2] = 0.9
        proj_mat[0, 3] = - x_translation
        proj_mat[1, 3] = y_translation
        if vertices_4_z is not None:
            vertices_4_z = self.transform_by_facewarehouse_param(vertices_4_z.unsqueeze(0))
            proj_mat[2, 3] = - vertices_4_z[:,:,2].mean()
        else:
            proj_mat[2, 3] = 32
        return proj_mat

    def transform_by_facewarehouse_param_and_rotate(self, vertices, elevation, azimuth, show_forehead=False):

        # rotation mat
        if elevation == 0 and azimuth == 180:
            rot_mat = self.default_rotation_mat
        else:
            rot_mat = self.get_rotation_mat(elevation, azimuth)

        trans_mat = torch.cat((self.trans_mat, torch.from_numpy(numpy.array([0,0,0,1])).cuda().float().unsqueeze(0)), dim=0)
        total_mat = rot_mat.mm(self.fake_proj_mat.mm(trans_mat))
        # total_mat = self.fake_proj_mat.mm(trans_mat)


        # proj_trans_mat = torch.matmul(proj_mat, trans_mat)
        vertices = torch.cat((vertices, torch.ones_like(vertices[:,:,1]).unsqueeze(2)),dim=2)
        vertices = torch.matmul(total_mat, vertices.permute((0,2,1))).permute(0,2,1)
        # print('z_min', vertices[:,:,2].min())
        # print('z_max', vertices[:,:,2].max())

        # z = vertices[:, :, 2]
        # x = vertices[:, :, 0] * (1.5-z)
        # y = vertices[:, :, 1] * (1.5-z)
        #
        # vertices = torch.stack((x, y, z), dim=2)



        if show_forehead:
            y_max = vertices[:, :, 1].max()
            y_min = vertices[:, :, 1].min()
            y_diff = (y_max - y_min)/2
            # print('y_min', vertices[:,:,1].min())
            # print('y_max', vertices[:,:,1].max())
            y = vertices[:, :, 1]/y_diff
            y = y - y_max/y_diff + 1
            z = (vertices[:, :, 2] - vertices[:, :, 2].max()) / y_diff
            x = vertices[:, :, 0]/y_diff
        else:
            z = vertices[:, :, 2] - vertices[:, :, 2].max()
            y = vertices[:, :, 1]
            x = vertices[:, :, 0]

        vertices = torch.stack((x, y, z), dim=2)

        return vertices

    def rotate_ortho(self, vertices, elevation, azimuth, show_forehead=False, fix_width=False):

        vertices0 = vertices
        if fix_width and self.fix_width_left == self.fix_width_right:
            vertices = torch.matmul(self.default_rotation_mat, vertices.permute((0,2,1))).permute(0,2,1)
            y_max = vertices[:, :, 1].max()
            y_min = vertices[:, :, 1].min()
            y_diff = (y_max - y_min)/2
            # print('y_min', vertices[:,:,1].min())
            # print('y_max', vertices[:,:,1].max())
            y = vertices[:, :, 1]/y_diff
            y = y - y_max/y_diff + 1
            z = (vertices[:, :, 2] - vertices[:, :, 2].max()) / y_diff
            x = vertices[:, :, 0]/y_diff
            x_diff = (x.max() - x.min())/2
            x = x - x.max() + x_diff
            self.fix_width_left = x.min()
            self.fix_width_right = x.max()
            print(self.fix_width_left, self.fix_width_right)
        vertices = vertices0
        
        # rotation mat
        if elevation == 0 and azimuth == 180:
            rot_mat = self.default_rotation_mat
        else:
            rot_mat = self.get_rotation_mat(elevation, azimuth)

        vertices = torch.matmul(rot_mat, vertices.permute((0,2,1))).permute(0,2,1)
        # print('z_min', vertices[:,:,2].min())
        # print('z_max', vertices[:,:,2].max())

        if show_forehead:
            y_max = vertices[:, :, 1].max()
            y_min = vertices[:, :, 1].min()
            y_diff = (y_max - y_min)/2
            # print('y_min', vertices[:,:,1].min())
            # print('y_max', vertices[:,:,1].max())
            y = vertices[:, :, 1]/y_diff
            y = y - y_max/y_diff + 1
            z = (vertices[:, :, 2] - vertices[:, :, 2].max()) / y_diff
            x = vertices[:, :, 0]/y_diff
            x_diff = (x.max() - x.min())/2
            x = x - x.max() + x_diff
            if fix_width:
                x_diff = x.max() - x.min()
                width_fix = self.fix_width_right - self.fix_width_left
                x = x / x_diff * width_fix
                x = x - x.min() + self.fix_width_left
                y = y / x_diff * width_fix
                z = z / x_diff * width_fix
                
        else:
            z = vertices[:, :, 2] - vertices[:, :, 2].max()
            y = vertices[:, :, 1]
            x = vertices[:, :, 0]

        vertices = torch.stack((x, y, z), dim=2)

        return vertices
