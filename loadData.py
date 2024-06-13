import os

import face_recognition


def loadDataF(dir_path):

    data = {}

    for file in os.listdir(dir_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(dir_path, file)
            image = face_recognition.load_image_file(img_path)
            face_encoding = face_recognition.face_encodings(image)

            if face_encoding:
                face_encoding = face_encoding[0]
                name = os.path.splitext(file)[0]
                data[name] = face_encoding

    return data