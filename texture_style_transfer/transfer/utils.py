import os

import imageio
import numpy as np
import cv2


def get_uv_landmark_from_landmark_vertex(obj_file, landmark_vertex_id):
    # not sure why this one needs to +1
    landmark_vertex_id = landmark_vertex_id + 1
    with open(obj_file) as f:
        lines = f.readlines()
    landmark_texture_uv = []
    uv_start_id = 0
    found_ud_start_id = False
    for id in landmark_vertex_id:
        id = str(id)
        found_this_one = False
        for i, line in enumerate(lines):
            if len(line.split()) == 0:
                continue
            if found_ud_start_id == False and line.split()[0] == 'vt':
                uv_start_id = i
                found_ud_start_id = True
            three_points = line.split()[1:]
            for point in three_points:
                if point.split('/')[0] == id:
                    # uv_id = int(point.split('/')[1])
                    uv_id = int(point.split('/')[1]) - 1
                    uv_id = uv_start_id + uv_id
                    uv = lines[uv_id].split()[1:]
                    landmark_texture_uv.append(uv)
                    found_this_one = True
                    break
            if found_this_one:
                break
    # print(landmark_texture_uv)
    landmarks = np.array(landmark_texture_uv).astype(float)
    # landmarks *= 256
    landmarks[:, 1] = 1 - landmarks[:, 1]
    return landmarks


def get_mask_from_landmark(landmarks, dilate_iter, texture_width, texture_height, manual_uv_ld=None):
    # left_eyebow_landmarks = np.array([[landmarks[17, 0], landmarks[17, 1]],
    #                                   [landmarks[18, 0], landmarks[18, 1]],
    #                                   [landmarks[19, 0], landmarks[19, 1]],
    #                                   [landmarks[20, 0], landmarks[20, 1]],
    #                                   [landmarks[21, 0], landmarks[21, 1]]]).astype(int)
    # right_eyebow_landmarks = np.array([[landmarks[22, 0], landmarks[22, 1]],
    #                                    [landmarks[23, 0], landmarks[23, 1]],
    #                                    [landmarks[24, 0], landmarks[24, 1]],
    #                                    [landmarks[25, 0], landmarks[25, 1]],
    #                                    [landmarks[26, 0], landmarks[26, 1]]]).astype(int)
    # left_eye_landmarks = np.array([[landmarks[36, 0], landmarks[36, 1]],
    #                                [landmarks[37, 0], landmarks[37, 1]],
    #                                [landmarks[38, 0], landmarks[38, 1]],
    #                                [landmarks[39, 0], landmarks[39, 1]],
    #                                [landmarks[40, 0], landmarks[40, 1]],
    #                                [landmarks[41, 0], landmarks[41, 1]]]).astype(int)
    # right_eye_landmarks = np.array([[landmarks[42, 0], landmarks[42, 1]],
    #                                 [landmarks[43, 0], landmarks[43, 1]],
    #                                 [landmarks[44, 0], landmarks[44, 1]],
    #                                 [landmarks[45, 0], landmarks[45, 1]],
    #                                 [landmarks[46, 0], landmarks[46, 1]],
    #                                 [landmarks[47, 0], landmarks[47, 1]]]).astype(int)
    if manual_uv_ld is not None:
        left_eye_landmarks = np.array([[manual_uv_ld[0, 0], manual_uv_ld[0, 1]],
                                       [manual_uv_ld[1, 0], manual_uv_ld[1, 1]],
                                       [manual_uv_ld[2, 0], manual_uv_ld[2, 1]],
                                       [manual_uv_ld[3, 0], manual_uv_ld[3, 1]],
                                       [manual_uv_ld[4, 0], manual_uv_ld[4, 1]],
                                       [manual_uv_ld[5, 0], manual_uv_ld[5, 1]],
                                       [manual_uv_ld[6, 0], manual_uv_ld[6, 1]],
                                       [manual_uv_ld[7, 0], manual_uv_ld[7, 1]]]).astype(int)
        right_eye_landmarks = np.array([[512 - manual_uv_ld[0, 0], manual_uv_ld[0, 1]],
                                        [512 - manual_uv_ld[1, 0], manual_uv_ld[1, 1]],
                                        [512 - manual_uv_ld[2, 0], manual_uv_ld[2, 1]],
                                        [512 - manual_uv_ld[3, 0], manual_uv_ld[3, 1]],
                                        [512 - manual_uv_ld[4, 0], manual_uv_ld[4, 1]],
                                        [512 - manual_uv_ld[5, 0], manual_uv_ld[5, 1]],
                                        [512 - manual_uv_ld[6, 0], manual_uv_ld[6, 1]],
                                        [512 - manual_uv_ld[7, 0], manual_uv_ld[7, 1]]]).astype(int)
        nose_landmarks = np.array([[manual_uv_ld[8, 0], manual_uv_ld[8, 1]],
                                   [manual_uv_ld[9, 0], manual_uv_ld[9, 1]],
                                   [manual_uv_ld[10, 0], manual_uv_ld[10, 1]],
                                   [manual_uv_ld[11, 0], manual_uv_ld[11, 1]],
                                   [manual_uv_ld[12, 0], manual_uv_ld[12, 1]]]).astype(int)
        mouth_landmarks = np.array([[manual_uv_ld[13, 0], manual_uv_ld[13, 1]],
                                    [manual_uv_ld[14, 0], manual_uv_ld[14, 1]],
                                    [manual_uv_ld[15, 0], manual_uv_ld[15, 1]],
                                    [manual_uv_ld[16, 0], manual_uv_ld[16, 1]],
                                    [manual_uv_ld[17, 0], manual_uv_ld[17, 1]],
                                    [manual_uv_ld[18, 0], manual_uv_ld[18, 1]],
                                    [manual_uv_ld[19, 0], manual_uv_ld[19, 1]],
                                    [manual_uv_ld[20, 0], manual_uv_ld[20, 1]]]).astype(int)
    else:
        left_eye_landmarks = np.array([[landmarks[17, 0], landmarks[17, 1]],
                                       [landmarks[18, 0], landmarks[18, 1]],
                                       [landmarks[19, 0], landmarks[19, 1]],
                                       [landmarks[20, 0], landmarks[20, 1]],
                                       [landmarks[21, 0], landmarks[21, 1]],
                                       [landmarks[39, 0], landmarks[39, 1]],
                                       [landmarks[40, 0], landmarks[40, 1]],
                                       [landmarks[41, 0], landmarks[41, 1]],
                                       [landmarks[36, 0], landmarks[36, 1]]]).astype(int)
        right_eye_landmarks = np.array([[landmarks[22, 0], landmarks[22, 1]],
                                        [landmarks[23, 0], landmarks[23, 1]],
                                        [landmarks[24, 0], landmarks[24, 1]],
                                        [landmarks[25, 0], landmarks[25, 1]],
                                        [landmarks[26, 0], landmarks[26, 1]],
                                        [landmarks[45, 0], landmarks[45, 1]],
                                        [landmarks[46, 0], landmarks[46, 1]],
                                        [landmarks[47, 0], landmarks[47, 1]],
                                        [landmarks[42, 0], landmarks[42, 1]]]).astype(int)
        nose_landmarks = np.array([[landmarks[27, 0], landmarks[27, 1]],
                                   [landmarks[31, 0], landmarks[31, 1]],
                                   [landmarks[32, 0], landmarks[32, 1]],
                                   [landmarks[33, 0], landmarks[33, 1]],
                                   [landmarks[34, 0], landmarks[34, 1]],
                                   [landmarks[35, 0], landmarks[35, 1]]]).astype(int)
        mouth_landmarks = np.array([[landmarks[48, 0], landmarks[48, 1]],
                                    [landmarks[49, 0], landmarks[49, 1]],
                                    [landmarks[50, 0], landmarks[50, 1]],
                                    [landmarks[51, 0], landmarks[51, 1]],
                                    [landmarks[52, 0], landmarks[52, 1]],
                                    [landmarks[53, 0], landmarks[53, 1]],
                                    [landmarks[54, 0], landmarks[54, 1]],
                                    [landmarks[55, 0], landmarks[55, 1]],
                                    [landmarks[56, 0], landmarks[56, 1]],
                                    [landmarks[57, 0], landmarks[57, 1]],
                                    [landmarks[58, 0], landmarks[58, 1]],
                                    [landmarks[59, 0], landmarks[59, 1]]]).astype(int)
    face_landmarks = np.array([[landmarks[0, 0], 0],
                               [landmarks[0, 0], landmarks[0, 1]],
                               [landmarks[1, 0], landmarks[1, 1]],
                               [landmarks[2, 0], landmarks[2, 1]],
                               [landmarks[3, 0], landmarks[3, 1]],
                               [landmarks[4, 0], landmarks[4, 1]],
                               [landmarks[5, 0], landmarks[5, 1]],
                               [landmarks[6, 0], landmarks[6, 1]],
                               [landmarks[7, 0], landmarks[7, 1]],
                               [landmarks[8, 0], landmarks[8, 1]],
                               [landmarks[9, 0], landmarks[9, 1]],
                               [landmarks[10, 0], landmarks[10, 1]],
                               [landmarks[11, 0], landmarks[11, 1]],
                               [landmarks[12, 0], landmarks[12, 1]],
                               [landmarks[13, 0], landmarks[13, 1]],
                               [landmarks[14, 0], landmarks[14, 1]],
                               [landmarks[15, 0], landmarks[15, 1]],
                               [landmarks[16, 0], landmarks[16, 1]],
                               [landmarks[16, 0], 0]]).astype(int)

    kernel = np.ones((3, 3), np.uint8)

    mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    mask_left_eye_poly = cv2.fillPoly(mask_background, [left_eye_landmarks], 1)
    mask_left_eye_poly = cv2.dilate(mask_left_eye_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    mask_right_eye_poly = cv2.fillPoly(mask_background, [right_eye_landmarks], 1)
    mask_right_eye_poly = cv2.dilate(mask_right_eye_poly, kernel, iterations=dilate_iter)[None, :, :]
    # mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    # mask_left_eyebow_poly = cv2.fillPoly(mask_background, [left_eyebow_landmarks], 1)
    # mask_left_eyebow_poly = cv2.dilate(mask_left_eyebow_poly, kernel, iterations=dilate_iter)[None, :, :]
    # mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    # mask_right_eyebow_poly = cv2.fillPoly(mask_background, [right_eyebow_landmarks], 1)
    # mask_right_eyebow_poly = cv2.dilate(mask_right_eyebow_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    mask_nose_poly = cv2.fillPoly(mask_background, [nose_landmarks], 1)
    mask_nose_poly = cv2.dilate(mask_nose_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    mask_mouth_poly = cv2.fillPoly(mask_background, [mouth_landmarks], 1)
    mask_mouth_poly = cv2.dilate(mask_mouth_poly, kernel, iterations=dilate_iter)[None, :, :]

    mask_background = np.zeros((texture_height, texture_width, 1), np.uint8)
    mask_face_all = cv2.fillPoly(mask_background, [face_landmarks], 1)
    mask_face_all = cv2.dilate(mask_face_all, kernel, iterations=dilate_iter)[None, :, :]
    mask_face_else = mask_face_all - np.clip((mask_left_eye_poly + mask_right_eye_poly + mask_nose_poly + mask_mouth_poly),
                                       0, 1)

    mask_to_show = 1 - mask_face_else
    mask_to_show = np.clip(mask_to_show, 0.5, 1)
    masks_region = np.concatenate(
        (mask_left_eye_poly, mask_right_eye_poly, mask_nose_poly, mask_mouth_poly, mask_face_else, mask_face_all))

    return masks_region, mask_to_show


def add_landmark(image, lm_coordinate, is_target=False):
    lm_num = lm_coordinate.shape[0]
    for i in range(0, lm_num):
        x = np.int(lm_coordinate[i, 0])
        y = np.int(lm_coordinate[i, 1])
        if is_target:
            image[y - 1:y + 2, x - 1:x + 2, :] = [0, 255, 0]
        else:
            image[y - 1:y + 2, x - 1:x + 2, :] = [255, 0, 0]
    return image


def get_facewarehouse_param(proj_param_file, trans_param_file):

    proj_param = []
    trans_param = []

    with open(proj_param_file) as f:
        lines = f.readlines()
        for line in lines:
            params = line.split()
            for param in params:
                proj_param.append(float(param))

    with open(trans_param_file) as f:
        lines = f.readlines()
        for line in lines:
            params = line.split()
            for param in params:
                trans_param.append(float(param))

    return np.array(proj_param), np.array(trans_param)


def get_landmarks_from_file(filename, start_id=3, is_DP_format=False):
    # style_landmark_file = 'data/new_styles/' + os.path.basename(filename)[:-4] + '.txt'
    landmarks_style = np.zeros((68, 2))

    if not is_DP_format:
        with open(filename) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[start_id:]):
                landmark = line.split()[:2]
                landmarks_style[i] = np.array(landmark, dtype='float32')
    else:
        with open(filename) as f:
            lines = f.readlines()
            for i, line in enumerate(lines[1:]):
                landmark = line.split()[1:3]
                landmarks_style[i] = np.array(landmark, dtype='float32')

    return landmarks_style


def generate_mask(landmarks, save_path='', save=True, image_size=512):

    dilate_iter = 3

    left_eye_landmarks = np.array([[landmarks[17, 0], landmarks[17, 1]],
                                       [landmarks[18, 0], landmarks[18, 1]],
                                       [landmarks[19, 0], landmarks[19, 1]],
                                       [landmarks[20, 0], landmarks[20, 1]],
                                       [landmarks[21, 0], landmarks[21, 1]],
                                       [landmarks[39, 0], landmarks[39, 1]],
                                       [landmarks[40, 0], landmarks[40, 1]],
                                       [landmarks[41, 0], landmarks[41, 1]],
                                       [landmarks[36, 0], landmarks[36, 1]]]).astype(int)
    right_eye_landmarks = np.array([[landmarks[22, 0], landmarks[22, 1]],
                                        [landmarks[23, 0], landmarks[23, 1]],
                                        [landmarks[24, 0], landmarks[24, 1]],
                                        [landmarks[25, 0], landmarks[25, 1]],
                                        [landmarks[26, 0], landmarks[26, 1]],
                                        [landmarks[45, 0], landmarks[45, 1]],
                                        [landmarks[46, 0], landmarks[46, 1]],
                                        [landmarks[47, 0], landmarks[47, 1]],
                                        [landmarks[42, 0], landmarks[42, 1]]]).astype(int)
    nose_landmarks = np.array([[landmarks[27, 0], landmarks[27, 1]],
                                   [landmarks[31, 0], landmarks[31, 1]],
                                   [landmarks[32, 0], landmarks[32, 1]],
                                   [landmarks[33, 0], landmarks[33, 1]],
                                   [landmarks[34, 0], landmarks[34, 1]],
                                   [landmarks[35, 0], landmarks[35, 1]]]).astype(int)
    mouth_landmarks = np.array([[landmarks[48, 0], landmarks[48, 1]],
                                    [landmarks[49, 0], landmarks[49, 1]],
                                    [landmarks[50, 0], landmarks[50, 1]],
                                    [landmarks[51, 0], landmarks[51, 1]],
                                    [landmarks[52, 0], landmarks[52, 1]],
                                    [landmarks[53, 0], landmarks[53, 1]],
                                    [landmarks[54, 0], landmarks[54, 1]],
                                    [landmarks[55, 0], landmarks[55, 1]],
                                    [landmarks[56, 0], landmarks[56, 1]],
                                    [landmarks[57, 0], landmarks[57, 1]],
                                    [landmarks[58, 0], landmarks[58, 1]],
                                    [landmarks[59, 0], landmarks[59, 1]]]).astype(int)
    face_landmarks = np.array([[landmarks[0, 0], 0],
                               [landmarks[0, 0], landmarks[0, 1]],
                               [landmarks[1, 0], landmarks[1, 1]],
                               [landmarks[2, 0], landmarks[2, 1]],
                               [landmarks[3, 0], landmarks[3, 1]],
                               [landmarks[4, 0], landmarks[4, 1]],
                               [landmarks[5, 0], landmarks[5, 1]],
                               [landmarks[6, 0], landmarks[6, 1]],
                               [landmarks[7, 0], landmarks[7, 1]],
                               [landmarks[8, 0], landmarks[8, 1]],
                               [landmarks[9, 0], landmarks[9, 1]],
                               [landmarks[10, 0], landmarks[10, 1]],
                               [landmarks[11, 0], landmarks[11, 1]],
                               [landmarks[12, 0], landmarks[12, 1]],
                               [landmarks[13, 0], landmarks[13, 1]],
                               [landmarks[14, 0], landmarks[14, 1]],
                               [landmarks[15, 0], landmarks[15, 1]],
                               [landmarks[16, 0], landmarks[16, 1]],
                               [landmarks[16, 0], 0]]).astype(int)

    kernel = np.ones((3, 3), np.uint8)

    mask_background = np.zeros((image_size, image_size, 1), np.uint8)
    mask_left_eye_poly = cv2.fillPoly(mask_background, [left_eye_landmarks], 1)
    mask_left_eye_poly = cv2.dilate(mask_left_eye_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((image_size, image_size, 1), np.uint8)
    mask_right_eye_poly = cv2.fillPoly(mask_background, [right_eye_landmarks], 1)
    mask_right_eye_poly = cv2.dilate(mask_right_eye_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((image_size, image_size, 1), np.uint8)
    mask_nose_poly = cv2.fillPoly(mask_background, [nose_landmarks], 1)
    mask_nose_poly = cv2.dilate(mask_nose_poly, kernel, iterations=dilate_iter)[None, :, :]
    mask_background = np.zeros((image_size, image_size, 1), np.uint8)
    mask_mouth_poly = cv2.fillPoly(mask_background, [mouth_landmarks], 1)
    mask_mouth_poly = cv2.dilate(mask_mouth_poly, kernel, iterations=dilate_iter)[None, :, :]

    mask_background = np.zeros((image_size, image_size, 1), np.uint8)
    mask_face_all = cv2.fillPoly(mask_background, [face_landmarks], 1)
    mask_face_all = cv2.dilate(mask_face_all, kernel, iterations=dilate_iter)[None, :, :]
    mask_face_else = np.clip(mask_face_all - np.clip((mask_left_eye_poly + mask_right_eye_poly + mask_nose_poly + mask_mouth_poly),
                                       0, 1), 0, 1)

    mask_cat = np.zeros((image_size, image_size, 1), np.uint8).transpose(2, 0, 1)
    mask_cat = np.clip((mask_cat + mask_left_eye_poly*50 + mask_right_eye_poly*100), 0, 100)
    mask_cat = np.clip((mask_cat + mask_nose_poly*150), 0, 150)
    mask_cat = np.clip((mask_cat + mask_mouth_poly*200), 0, 200)
    mask_cat = np.clip((mask_cat + mask_face_else * 250), 0, 250)

    mask_cat = np.repeat(mask_cat,3,axis=0).transpose(1, 2, 0)

    if save:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        imageio.imsave(save_path, mask_cat.astype(np.uint8))
    else:
        # imageio.imsave('check_temp.jpg', mask_cat.astype(np.uint8))
        return mask_cat/255

def erase_background(image_path, ladnmark_path, path_style_masked, path_mask_style=None, save=True):

    image = imageio.imread(image_path)[:,:,0:3]
    landmarks = get_landmarks_from_file(ladnmark_path, start_id=0)
    dilate_iter = 2
    kernel = np.ones((3, 3), np.uint8)

    face_landmarks = np.array([[landmarks[0, 0], 0],
                               [landmarks[0, 0], landmarks[0, 1]],
                               [landmarks[1, 0], landmarks[1, 1]],
                               [landmarks[2, 0], landmarks[2, 1]],
                               [landmarks[3, 0], landmarks[3, 1]],
                               [landmarks[4, 0], landmarks[4, 1]],
                               [landmarks[5, 0], landmarks[5, 1]],
                               [landmarks[6, 0], landmarks[6, 1]],
                               [landmarks[7, 0], landmarks[7, 1]],
                               [landmarks[8, 0], landmarks[8, 1]],
                               [landmarks[9, 0], landmarks[9, 1]],
                               [landmarks[10, 0], landmarks[10, 1]],
                               [landmarks[11, 0], landmarks[11, 1]],
                               [landmarks[12, 0], landmarks[12, 1]],
                               [landmarks[13, 0], landmarks[13, 1]],
                               [landmarks[14, 0], landmarks[14, 1]],
                               [landmarks[15, 0], landmarks[15, 1]],
                               [landmarks[16, 0], landmarks[16, 1]],
                               [landmarks[16, 0], 0]]).astype(int)
    
#     face_landmarks = np.array([[landmarks[26, 0], landmarks[26, 1]],
#                                    [landmarks[25, 0], landmarks[25, 1]],
#                                    [landmarks[24, 0], landmarks[24, 1]],
#                                    [landmarks[23, 0], landmarks[23, 1]],
#                                    [landmarks[22, 0], landmarks[22, 1]],
#                                    [landmarks[21, 0], landmarks[21, 1]],
#                                    [landmarks[20, 0], landmarks[20, 1]],
#                                    [landmarks[19, 0], landmarks[19, 1]],
#                                    [landmarks[18, 0], landmarks[18, 1]],
#                                    [landmarks[17, 0], landmarks[17, 1]],
#                                [landmarks[0, 0], landmarks[0, 1]],
#                                [landmarks[1, 0], landmarks[1, 1]],
#                                [landmarks[2, 0], landmarks[2, 1]],
#                                [landmarks[3, 0], landmarks[3, 1]],
#                                [landmarks[4, 0], landmarks[4, 1]],
#                                [landmarks[5, 0], landmarks[5, 1]],
#                                [landmarks[6, 0], landmarks[6, 1]],
#                                [landmarks[7, 0], landmarks[7, 1]],
#                                [landmarks[8, 0], landmarks[8, 1]],
#                                [landmarks[9, 0], landmarks[9, 1]],
#                                [landmarks[10, 0], landmarks[10, 1]],
#                                [landmarks[11, 0], landmarks[11, 1]],
#                                [landmarks[12, 0], landmarks[12, 1]],
#                                [landmarks[13, 0], landmarks[13, 1]],
#                                [landmarks[14, 0], landmarks[14, 1]],
#                                [landmarks[15, 0], landmarks[15, 1]],
#                                [landmarks[16, 0], landmarks[16, 1]]]).astype(int)

    mask_background = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    mask_face_all = cv2.fillPoly(mask_background, [face_landmarks], 1)
    mask_face_all = cv2.dilate(mask_face_all, kernel, iterations=dilate_iter)[:, :, None]

    image = image * mask_face_all + np.ones_like(image) * 255 * (1-mask_face_all)
    save_dir = os.path.dirname(path_style_masked)
    os.makedirs(save_dir, exist_ok=True)
    imageio.imsave(path_style_masked, image.astype(np.uint8))
    
    if path_mask_style is not None:
        mask_style_image = np.ones_like(image) * 255 * (1-mask_face_all)
        save_dir = os.path.dirname(path_mask_style)
        os.makedirs(save_dir, exist_ok=True)
        imageio.imsave(path_mask_style, mask_style_image.astype(np.uint8))
        return mask_style_image 

def transform_landmarks_to_uncrop(small_style, small_content, big_content):

    small_content_3pts = np.float32(small_content[[17, 26, 30], :])
    big_content_3pts = np.float32(big_content[[17, 26, 30], :])
    M = cv2.getAffineTransform(small_content_3pts, big_content_3pts)

    # print('transform error: ', np.mean(np.matmul(M, np.concatenate((small_content, np.ones_like(small_content[:, 0:1])), axis=1).transpose(1,0))[:,:40] - big_content[:40].transpose(1,0)))

    small_style = np.concatenate((small_style, np.ones_like(small_style[:, 0:1])), axis=1).transpose(1,0)
    big_style = np.matmul(M, small_style).transpose(1,0)
    return big_style

def save_landmarks(landmarks, save_path):

    with open(save_path, 'w') as f:
        for landmark in landmarks:
            f.write('%.8f %.8f\n' % (landmark[0], landmark[1]))
