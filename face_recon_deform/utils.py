
import imageio
import numpy as np
import cv2

def get_landmarks_from_file(filename, start_row=0, start_col=0):
    with open(filename) as f:
        lines = f.readlines()
        landmarks_style = np.zeros((lines.__len__()-start_row, lines[start_row].split().__len__()-start_col))
        for i, line in enumerate(lines[start_row:]):
            landmark = line.split()[start_col:]
            landmarks_style[i] = np.array(landmark, dtype='float32')
    return landmarks_style

def save_landmarks(landmarks, save_path):
    with open(save_path, 'w') as f:
        for landmark in landmarks:
            f.write('%.8f %.8f\n' % (landmark[0], landmark[1]))

def transform_landmarks_to_uncrop(small_style, small_content, big_content):
    small_content_3pts = np.float32(small_content[[17, 26, 30], :])
    big_content_3pts = np.float32(big_content[[17, 26, 30], :])
    M = cv2.getAffineTransform(small_content_3pts, big_content_3pts)

    small_style = np.concatenate((small_style, np.ones_like(small_style[:, 0:1])), axis=1).transpose(1,0)
    big_style = np.matmul(M, small_style).transpose(1,0)
    return big_style

def pre_process_FFHQ(image_path, landmark_path, save_image_path, save_landmark_path):
    image = imageio.imread(image_path)
    image_square = image[100:-100,:,:]
    imageio.imwrite(save_image_path, image_square)

    landmarks = get_landmarks_from_file(landmark_path)
    landmarks[:, 1] = landmarks[:, 1] - 100
    save_landmarks(landmarks, save_landmark_path)

