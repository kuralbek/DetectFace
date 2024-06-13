import cv2
import face_recognition
import numpy as np

def detectRecFace(frame, knowFace,face_cascade):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # rgb_frame = small_frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)




    face_locations = face_recognition.face_locations(rgb_frame)

    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
        matches = face_recognition.compare_faces(list(knowFace.values()), face_encoding)
        name = "Unknown"


        face_distances = face_recognition.face_distance(list(knowFace.values()), face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list(knowFace.keys())[best_match_index]
            print('name',name)


        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), ( right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 255, 0, 255), 1)

    return frame
