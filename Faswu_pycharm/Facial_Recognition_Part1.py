#샘플 모으는 코드
import sys
import cv2
import os
import subprocess

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#디렉터리 나누기 추가
def createFolder(dir):
    try:
        sys.setrecursionlimit(2000)
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error:Creating directory.' + dir)


def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

cap = cv2.VideoCapture(0)
count = 0
c=0
c+=1
file_list = os.listdir('knowns/')
max = 1

for file in file_list:
    index = int(file[4:])
    print("facial_index: ",index)
    if index >= max :
        max = index + 1
    print("facial_max: ",max)
print(max)
createFolder('C:/faswu-master/faswu-master/Faswu_pycharm/knowns/user'+str(max))

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'knowns/user'+str(max)+'/'+'pic'+str(count)+'_user'+str(max)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==50:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')

os.system('python "C:/faswu-master/faswu-master/Faswu_pycharm/face_recog.py"')


