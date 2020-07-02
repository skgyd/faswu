# face_recog.py
# 샘플 모은 후 인식하는 코드
# 모듈불러오기
from __future__ import print_function
from connect import Connect
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import threading
from PyQt5 import QtCore
from PyQt5 import uic
import socket
import sys
from PyQt5.QtWidgets import *  # PyQt import
from PyQt5.QtGui import *
import cv2 as cv
import face_recognition
import cv2
import camera
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
unKnown_box = []
Known_box = []
global user_privacy

Input_users = []
global num
past_privacy_list = []
user_privacy = []
unKnown_privacy = True
event = True
input_user = ''
count = 0

from tkinter import *
from tkinter import ttk
from tkinter import messagebox

import shutil

import os.path  

file = 'C:/Users/user_name/PycharmProjects /project_name/input_users.txt'
file2 =  'C:/Users/user_name/PycharmProjects/project_name/user_privacy.txt'
if os.path.isfile(file):
    file = open( 'C:/Users/user_name/PycharmProjects/project_name/input_users.txt', 'r')
    lines = file.readlines()
    line = file.readline()

    lines = [line.rstrip('\n') for line in lines]

    file.close()

    for line in lines:
        Input_users.append(line)

    print(Input_users)

    os.remove('C:/Users/user_name/PycharmProjects/project_name/input_users.txt')

else:
    print("Input users: empty")

if os.path.isfile(file2):
    file = open('C:/Users/user_name/PycharmProjects/project_name/user_privacy.txt', 'r')
    lines = file.readlines()
    line = file.readline()

    lines = [line.rstrip('\n') for line in lines]
    file.close()

    for line in lines:
        past_privacy_list.append(line)

    changes = {"True": True, "False": False}
    user_privacy = [changes.get(x, False) for x in past_privacy_list]

    os.remove('C:/Users/user_name/PycharmProjects/project_name/user_privacy.txt')

else:
    user_privacy = [False] * 100

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()
        self.known_face_encodings = []
        self.known_face_names = []
        # Load sample pictures and learn how to recognize it.

        # knowns 디렉토리에서 사진 파일을 읽습니다. 파일 이름으로부터 사람 이름을 추출합니다
        # user 폴더 추가
        file_list = os.listdir('knowns/')
        files = []
        for file in file_list:
            index = int(file[4:])
            dirname = 'knowns/user' + str(index)
            files = os.listdir(dirname)

            for filename in files:
                name, ext = os.path.splitext(filename)
                if ext == '.jpg':
                    self.known_face_names.append(name)
                    pathname = os.path.join(dirname, filename)
                    print(pathname)

                    # 사진에서 얼굴 영역을 알아내고, face landmarks라 불리는
                    # 68개 얼굴 특징의 위치를 분석한 데이터를 known_face_encodings에 저장합니다.
                    img = face_recognition.load_image_file(pathname)

                    # face_encoding = face_recognition.face_encodings(img)[0]
                    face_encodings = face_recognition.face_encodings(img)

                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        self.known_face_encodings.append(face_encoding)

            # Initialize some variables
            self.face_locations = []
            self.face_encodings = []
            self.face_names = []
            self.process_this_frame = True

    def __del__(self):
        del self.camera

    @property
    def get_frame(self):

        # Grab a single frame of video
        # 카메라로부터 frame을 읽어서 1/4 크기로 줄입니다. 이것은 계산양을 줄이기 위해서 입니다.
        global user_privacy
        global rgb_small_frame
        frame = self.camera.get_frame()
        frame = cv2.flip(frame, 1)
        try:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        except Exception as e:
            print(str(e))
        # Only process every other frame of video to save time
        # 계산 양을 더 줄이기 위해서 두 frame당 1번씩만 계산합니다.
        if self.process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            # 읽은 frame에서 얼굴 영역과 특징을 추출합니다.
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:

                # See if the face is a match for the known face(s)
                # Frame에서 추출한 얼굴 특징과 knowns에 있던 사진 얼굴의 특징을 비교하여,
                # (얼마나 비슷한지) 거리 척도로 환산합니다.
                # 거리(distance)가 가깝다는 (작다는) 것은 서로 비슷한 얼굴이라는 의미 입니다.

                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                # 실험상, 거리가 0.6 이면 다른 사람의 얼굴입니다. 이런 경우의 이름은 Unknown 입니다.
                # 거리가 0.6 이하이고, 최소값을 가진 사람의 이름을 찾습니다.
                name = "Unknown"
                if min_value < 0.45:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]
                    name = name[5:]

                # if name=="Unknown":
                #    exec(open(Facial_Recognition_Part1).read())
                self.face_names.append(name)
        self.process_this_frame = not self.process_this_frame


        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        #######

        # 찾은 사람의 얼굴 영역과 이름을 비디오 화면에 그립니다.
        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 5
            bottom *= 7
            left *= 2

            if name == "Unknown" and unKnown_privacy:
                # print("Known_Box: ", Known_box, "name: ", name)
                # Extract the region of the image that contains the face
                face_image = frame[top:bottom, left:right]
                # Blur the face image
                face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
                # Put the blurred face region back into the frame image
                frame[top:bottom, left:right] = face_image

            for user in Input_users:
                if name == user and user_privacy[int(user[4:])]:
                    # Extract the region of the image that contains the face
                    face_image = frame[top:bottom, left:right]
                    # Blur the face image
                    face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
                    # Put the blurred face region back into the frame image
                    frame[top:bottom, left:right] = face_image

            # Draw a box around the face 얼굴박스
            cv2.rectangle(frame, (left, top - 50), (right, bottom), (255, 255, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top + 6), font, 1.0, (0, 0, 255), 1)

        for (top, right, bottom, left) in unKnown_box:
            le = int(left * 1.01)
            ri = int(right * 0.96)
            to = int(top * 1.05)
            bo = int(bottom * 0.96)
            face_image = frame[to:bo, le:ri]
            face_image = cv2.GaussianBlur(face_image, (33, 33), 30)
            frame[to:bo, le:ri] = face_image
            unKnown_box.append([top, right, bottom, left])

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()



if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)

    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    vbox = QtWidgets.QVBoxLayout()
    label = QtWidgets.QLabel()
    lineEdit = QLineEdit()

    running = False


    def run():
        global running
        print("started..")
        while running:
            global frame
            frame = face_recog.get_frame

            if True:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                label.setPixmap(pixmap)
            else:
                QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
                print("cannot read frame.")
                break

    def server(ip, port):
        while True:
            buf = 512
            width = 640
            height = 480
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            addr = (ip, int(port))
            code = 'start'
            code = ('start' + (buf - len(code)) * 'a').encode('utf-8')
            s.sendto(code, addr)
            data = frame.tostring()
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            for i in range(0, len(data), buf):
                s.sendto(data[i:i + buf], addr)
            print('start server...')


    running = True
    th = threading.Thread(target=run)
    th.start()

    def stop():
        global running
        running = False
        print("stoped...")


    def onExit():
        response = messagebox.askokcancel("DELETE USERS", "저장한 user들을 지우시겠습니까?")
        # 저장된 user들을 user1을 제외하고 삭제한다.
        if response == True:

            num = len(os.walk('C:/Users/user_name/PycharmProjects/project_name/knowns').__next__()[1])
            print(num)

            for i in range(2, num + 1):
                shutil.rmtree('C:/Users/user_name/PycharmProjects/project_name/knowns/user' + str(i))
        print("exit")
        stop()


    def client(ip,port):
        addr = (ip, int(port))
        print('서버와 연결 중...')
        buf = 512
        width = 640
        height = 480
        code = b'start'
        num_of_chunks = width * height * 3 / buf
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(addr)
        s.bind(addr)
        print('연결 완료')
        while True:
                chunks = []
                start = False
                while len(chunks) < num_of_chunks:
                    chunk, _ = s.recvfrom(buf)
                    if start:
                        chunks.append(chunk)
                    elif chunk.startswith(code):
                        start = True

                byte_frame = b''.join(chunks)

                frame = np.frombuffer(
                    byte_frame, dtype=np.uint8).reshape(height, width, 3)

                cv.imshow('recv', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                   break
        s.close()


    def connect():
        win = Connect()
        win.setWindowTitle('FASWU')
        r = win.showModal()

        if r:
            global ip, port
            ip = win.ip.text()
            port = win.port.text()
            print(ip)
            print(port)
            con = win.con
            if con == 1:
                print('server clicked...')
                server(ip, port)
            elif con == 2:
                print('client clicked...')
                client(ip,port)


    def learn():
        changes = {"True": True, "False": False}
        user_privacy = [changes.get(x, False) for x in past_privacy_list]

        file = open('C:/Users/user_name/PycharmProjects/project_name/input_users.txt', 'w')
        for Input_user in Input_users:
            file.write(Input_user + "\n")
        file.close()

        file = open('C:/Users/user_name/PycharmProjects/project_name/user_privacy.txt', 'w')
        for i in past_privacy_list:
            file.write(i + "\n")
        file.close()

        import Facial_Recognition_Part1
        sys.exit()


    def blur():
        global input_user, user_privacy, past_privacy_list
        input_user = lineEdit.text()
        Input_users.append(input_user)
        print(input_user)
        global num
        num = int(input_user[4:])
        # action2 = ttk.Button(win, text="EXIT", command=win.destroy)
        # action2.place(x=50, y=50)
        print(Input_users)
        print('*******************************************************')
        user_privacy[num] = not user_privacy[num]

        for i in user_privacy:
            past_privacy_list.append(str(user_privacy[i]))

        # Bool -> str
        changes = {True: "True", False: "False"}
        past_privacy_list = [changes.get(x, False) for x in user_privacy]

        print(past_privacy_list)


    btn_con = QtWidgets.QPushButton("connect")
    btn_server = QtWidgets.QPushButton("server")
    btn_client = QtWidgets.QPushButton("Client")
    btn_connect = QtWidgets.QPushButton("Connect")
    btn_learn = QtWidgets.QPushButton("learn")
    vbox.addWidget(label)
    vbox.addWidget(btn_con)
    vbox.addWidget(btn_learn)
    vbox.addWidget(lineEdit)
    win.setLayout(vbox)
    win.show()
    btn_con.clicked.connect(connect)
    btn_learn.clicked.connect(learn)
    lineEdit.returnPressed.connect(blur)
    app.aboutToQuit.connect(onExit)
    sys.exit(app.exec_())
    print('<<<finish>>>')