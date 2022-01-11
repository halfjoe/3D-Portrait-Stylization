import os

for file in os.listdir("upload"):
    if file.endswith(".jpg"):
        print(file.rsplit('.', 1)[0])
        os.system('PhotoAvatarLib.exe ' + file.rsplit('.', 1)[0])

        fp = open(os.path.join('result', file.rsplit('.', 1)[0] + '.mtl'), "w")
        fp.write('newmtl material_1\nmap_Kd %s_face.jpg' % file.rsplit('.', 1)[0])
        fp.close()

        fp = open(os.path.join('result', file.rsplit('.', 1)[0] + '_face_fit_ortho.obj'), "r")
        fstr = fp.read()
        fp.close()

        fp = open(os.path.join('result', file.rsplit('.', 1)[0] + '_face_fit_ortho.obj'), "w")
        fp.write('mtllib %s.mtl\nusemtl material_1\n' % file.rsplit('.', 1)[0])
        fp.write(fstr)
        fp.close()

