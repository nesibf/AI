# python3 -m pip install face_recognition
# pip3 install pytube --break-system-packages
# pip3 install yt_dlp --break-system-packages
# pip3 install setuptools --break-system-packages
# pip3 install opencv-python --break-system-packages
# pip3 install py2app setuptools --break-system-packages
# pip3 install PyQt6 --break-system-packages
# pip3 install --upgrade py2app setuptools --break-system-packages

## 1. Create a setup.py file
# py2applet --make-setup hello_world.py
# py2applet --make-setup face_detection.py

## 2. Cleanup build directories
# rm -rf build dist

## 3. Build application with alias mode
# python3 setup.py py2app -A

## 4. Running application
# ./dist/hello_world.app/Contents/MacOS/hello_world
# ./dist/face_detection.app/Contents/MacOS/face_detection

## 5. Steps for iOS development
# create apple dev account, no need to pay membership
# in Xcode Settings > login with your dev acount
# select your project and verify sigining is set
# connect your iPhone on Mac and press "Run"

print("Hello, world!")
