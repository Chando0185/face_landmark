import cv2
import dlib

# video = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frame = cv2.imread('4.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
facedata=faceDetect.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=None, flags=None, minSize=None, maxSize=None)
for x3,y3,w3,h3 in facedata:
    overlay = frame.copy()
    output = frame.copy()
    cv2.rectangle(overlay, (x3,y3), (x3+w3, y3+h3), (0,0,255), -2)
    alpha=0.2
    # frame=cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)

faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    landmarks = predictor(gray, face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(frame, (x, y), 2, (0,255,0), -1)
filename='chando22.jpg'
cv2.imwrite(filename, frame)
cv2.imshow("Result", frame)
cv2.waitKey(0)

cv2.destroyAllWindows()










