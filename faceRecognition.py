import cv2                  # thư viện hỗ trợ xử lí hình ảnh
import numpy as np          # thư viện numpy hỗ trợ các thao tác chuyển đổi ma trận ảnh
import os                   # thư viện hỗ trợ các thao tác với file, hệ điều hành
import sqlite3              # thư viện hỗ trợ thao tác với cơ sở dữ liệu
# from PIL import Image       #thư viện hỗ trợ các thao tác với chuyển đổi, đọc ảnh
import dlib
# import face_recognition
import time

import pyttsx3                           #thư viện hỗ trợ giọng nói
ass = pyttsx3.init()                     #khởi tạo một trợ lí ảo
voice = ass.getProperty('voices')        # thiết lập thư viện giọng nói cho trợ lí ảo
ass.setProperty('voice', voice[1].id)    # thiết lập giọng nữ cho trợ lí ảo

def speak(audio):                           #khai báo một hàm speak
    print("[ASSISTANT]  " + str(audio))     #in lên terminal thông báo
    ass.say(audio)                          #phát lên âm thanh thông báo 
    ass.runAndWait()  

def main_faceRecognition():
    speak("we're going to recognize, please wait till the camera is initialized ...")
    print("[INFO] loading HOG Face Detector...")
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml') #load model nhận diện khuôn mặt
    hogFaceDetector = dlib.get_frontal_face_detector()

    recognizer = cv2.face.LBPHFaceRecognizer_create()       # load model train và dự đoán dữ liệu

    recognizer.read('training.yml') #đọc dữ liệu file đã huấn luyện

    def getProfile(id):
        conn = sqlite3.connect("dataBase.db")               #kết nối đến CSDL
        query = "SELECT * FROM People WHERE ID="+str(id)    # thiết lập câu lệnh truy cập database để lấy thông tin
        cursor = conn.execute(query)        # thiết lập con trỏ thực thi request
        
        profile =None
        for row in cursor: # trong mỗi cột gán bằng biết profile
            profile = row
        
        conn.close() #đóng
        return profile

    cap = cv2.VideoCapture(0)               # khởi tạo camera
    fontface = cv2.FONT_HERSHEY_SIMPLEX     # thiết lập font chữ
    
    print('[INFO] performing face detection with Dlib...')
    per = True 
    cur_sec = time.localtime().tm_sec
    pre_sec = time.localtime().tm_sec
    while True:
        ret, frame = cap.read()         # đọc khung hình trả về từ cam
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #chuyển đổi khung hình từ màu BRG sang định dạng đen trắng
        
        # faces = face_cascade.detectMultiScale(gray, 1.3,5)#nhận điện các khuôn mặt có trong khung hình đã chuyển đổi màu
        faces = hogFaceDetector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            

        # for (x,y,w,h) in faces: # trong mỗi khuôn mặt sẽ trả về 4 giá trị x,y,w,h
            # cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0), 2)#vẽ bao quanh khuôn mặt một hình vuông
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0), 2)
            
            # roi_gray = gray[y:y+h, x:x+w] #gán biến roi_gray có độ dài và rộng bằng hình vuông vừa vẽ
            roi_gray = gray[y1:y2, x1:x2]
            
            roi = cv2.resize(roi_gray, (224,224))# thay đổi kích thuơcs ảnh về 224x224
            roiNP = np.array(roi, 'uint8')# ma trận hóa ảnh với kiểu dữ liệu là uint8
            
            id, error = recognizer.predict(roiNP)# hàm predict sẽ trả về 2 giá trị dự đoán id người dùng và độ sai lệch
            error = round(error,1)
            print(f"[id] {id}  [error] {error}")# in lên màn hình kết quả
            if error < 68:# nếu sai số nhở hơn 68
                profile = getProfile(id)# lấy profile tương ứng với id trả về trong cơ sở dữ liệu
                
                if (profile != None):# nếu máy dự đoán được khuôn mặt và thỏa mãn sai số nhỏ
                    # cv2.putText(frame, ""+str(profile[1]), (x,y+h+30), fontface,1,(0,255,0), 2)# vẽ lên khung ảnh tên người dùng
                    cv2.putText(frame, ""+str(profile[1]), (x1, y2+30 ), fontface,1, (0,255,0), 2)
                # if per == True:
                #     speak('hi '+ str(profile[1]))
                #     per = False
                # cur_sec = time.localtime().tm_sec
                # if cur_sec - pre_sec == 3:
                #     per = True
                #     pre_sec = cur_sec
                # if cur_sec==0:
                #     pre_sec = 0
                # if cur_sec - pre_sec > 3 or pre_sec - cur_sec > 3:
                #     cur_sec= time.localtime().tm_sec
                #     pre_sec= time.localtime().tm_sec
            else:
                # cv2.putText(frame, "Unknown",(x,y+h+30),fontface, (0,255,0), 2) # nếu không thì hiện Unknown
                cv2.putText(frame, "Unknown",(x1,y2+30),fontface,1,  (0,255,0), 2)
                # speak('Unknown')

        cv2.imshow("Face Recognition",frame)# hiện lên khung hình đã được nhận diện 
        
        if ( cv2.waitKey(1) == 27):# nếu nhấn phím 'esc'
            cap.release()#tắt camera
            cv2.destroyAllWindows()# đóng mọi cửa sổ đã được tạo bởi chương trình
            break#thoát khỏi vòng lặp vô hạn
            
    cap.release()#tắt camera
    cv2.destroyAllWindows()#đóng mọi cửa sổ đã được tạo bởi chương trình
    speak('the process is stopped by administrator')#phát ra thông báo

if __name__ == '__main__':
    main_faceRecognition()# nếu chương trình này được chạy thì sẽ thực thi hàm main_faceRecognition
    speak('all done! have a nice day, sir!')#thông báo