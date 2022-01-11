
import os
from face_recon_deform.utils import *

dir_data = '../data_demo/'

def prepare_reconstruct():
    names_data = os.listdir(dir_data)
    names_data.sort()
    path_save_batch = os.path.join('PhotoAvatarLib_exe/recon.bat')
    with open(path_save_batch, 'w') as f:
        f.write('cd PhotoAvatarLib_exe\n')
        for name in names_data:
            image_path = dir_data + name + '/' + name
            f.write('PhotoAvatarLib.exe ../../%s\n' % image_path)
    print('Reconstruction batch file saved to ', path_save_batch)

def do_reconstruct():
    os.system('PhotoAvatarLib_exe\\recon.bat')

def prepare_deform():
    names_data = os.listdir(dir_data)
    names_data.sort()
    path_save_batch = os.path.join('LaplacianDeformerConsole/deform.bat')
    with open(path_save_batch, 'w') as f:
        f.write('cd LaplacianDeformerConsole\n')
        for name in names_data:
            dir_one_data = os.path.join(dir_data,name)
            path_landmark_content = os.path.join(dir_one_data,name+'.txt')
            landmarks_content = get_landmarks_from_file(path_landmark_content)
            path_landmark_translated = os.path.join(dir_one_data,name+'_translated.txt')
            landmarks_translated = get_landmarks_from_file(path_landmark_translated)
            landmark_id = get_landmarks_from_file('LaplacianDeformerConsole/feature_id.txt')

            width_ratio = (landmarks_translated[16,0] - landmarks_translated[0,0]) / (landmarks_content[16,0] - landmarks_content[0,0])
            height_ratio = (landmarks_translated[19,1] - landmarks_translated[8,1]) / (landmarks_content[19,1] - landmarks_content[8,1])
            shrink_ratio = np.sqrt((width_ratio**2 + height_ratio**2)/2)

            path_landmark_deform = os.path.join(dir_one_data,name+'_deform.txt')
            path_obj_recon = os.path.join(dir_one_data,name+'_face_fit_ortho.obj')
            v_start_id = 0
            with open(path_obj_recon) as f_readobj:
                lines = f_readobj.readlines()
                for i, line in enumerate(lines):
                    if line.split()[0] == 'v':
                        v_start_id = i
                        break

                # obj file must NOT have vn mixed with v
                with open(path_landmark_deform, 'w') as f_ld:
                    f_ld.write('68\n')
                    for i in range(68):
                        z_value = lines[v_start_id + int(landmark_id[i+1]) - 0].split()[3]
                        f_ld.write('%d %.3f %.3f %.3f\n' % (landmark_id[i+1], landmarks_translated[i,0], landmarks_translated[i,1], float(z_value)*shrink_ratio))

            path_obj_deform = os.path.join(dir_one_data,name+'_face_fit_ortho_deform.obj')
            prefix = '../'
            f.write('LaplacianDeformerConsole.exe %s data/topology.obj %s %s 1000 1.0000\n'%((prefix+path_obj_recon), (prefix+path_landmark_deform), (prefix+path_obj_deform)))
    print('Deformation batch file saved to ', path_save_batch)

def do_deform():
    os.system('LaplacianDeformerConsole\\deform.bat')

def complete_objs():
    names_data = os.listdir(dir_data)
    names_data.sort()
    for name in names_data:
        dir_one_data = os.path.join(dir_data,name)
        path_mtl = os.path.join(dir_one_data, name + '.mtl')
        name_texture = name+'_face.jpg'
        with open(path_mtl, 'w') as f:
            f.write('newmtl %s\n' % 'material_1')
            f.write('map_Kd %s\n' % name_texture)

        names_one_data = os.listdir(dir_one_data)
        names_one_data.sort()
        for name_one_data in names_one_data:
            if name_one_data[-4:] == '.obj':
                with open(os.path.join(dir_one_data, name_one_data)) as f:
                    lines = f.readlines()
                    with open(os.path.join(dir_one_data, name_one_data), 'w') as f:
                        f.write('mtllib %s\n' % os.path.basename(path_mtl))
                        f.write('usemtl material_1\n')
                        for line in lines:
                            f.write(line)

if __name__ == '__main__':

    prepare_reconstruct()
    do_reconstruct()
    prepare_deform()
    do_deform()
    complete_objs()
