import pathlib
import cv2
import face_recognition
import numpy as np
from pytube import YouTube
import os
import yt_dlp


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

VIDEO_URL = 'https://www.youtube.com/watch?v=eZMeG6o04eU'

def download_youtube_video(url, output_path):
    try:
        ydl_opts = {
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            print(f"Downloading video: {info_dict['title']}")
            return os.path.join(output_path, f"{info_dict['title']}.mp4")
    except Exception as e:
        print(f"An error occurred while downloading the video: {e}")
        return None

def detect_faces(video_path):
    if not os.path.isfile(video_path):
        print(f"Video file does not exist: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    while True:
        _, frame = cap.read()
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
    cap.release()

def main():
    video_filename = download_youtube_video(VIDEO_URL, '.')

    if video_filename:
        # Extract frames from video
        detect_faces(video_filename)

        print("Face recognition completed.")
    else:
        print("Failed to download the video. Exiting.")

if __name__ == "__main__":
    main()
