import cv2

from detRec_face import detectRecFace
from faceDetect import detect_face
from loadData import loadDataF


# def main():
#     cap = cv2.VideoCapture(0)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
#     while True:
#
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Ошибка при получении кадра!")
#             break
#
#         frame = detect_face(frame, face_cascade)
#
#         cv2.imshow("Окно", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cv2.destroyAllWindows()
#     cap.release()

def main():

    knowFace = loadDataF("DataSet")


    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось захватить кадр.")
            break

        frame = detectRecFace(frame,knowFace,face_cascade)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Face Detector')
    main()
