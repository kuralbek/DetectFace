import cv2

from faceDetect import detect_face


def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Ошибка при получении кадра!")
            break

        frame = detect_face(frame, face_cascade)

        cv2.imshow("Окно", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    print('Face Detector')
    main()
