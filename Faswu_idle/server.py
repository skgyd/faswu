import socket
import numpy as np
import cv2 as cv


#연결시 클라이언트의 ip, port
addr = ("0.0.0.0", 8080)
buf = 512
width = 640
height = 480
cap = cv.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)
code = 'start'
code = ('start' + (buf - len(code)) * 'a').encode('utf-8')


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            s.sendto(code, addr)
            data = frame.tostring()
            for i in range(0, len(data), buf):
                s.sendto(data[i:i+buf], addr)

        else:
            break
 
