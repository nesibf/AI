import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_eye_tree_eyeglasses.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalcatface.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalcatface_extended.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt2.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt_tree.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_fullbody.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_lefteye_2splits.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_license_plate_rus_16stages.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_lowerbody.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_profileface.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_righteye_2splits.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_russian_plate_number.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_smile.xml"
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_upperbody.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
