import cv2                  # thư viện hỗ trợ xử lí hình ảnh
import os                   # thư viện hỗ trợ các thao tác với file, hệ điều hành
import numpy as np          # thư viện numpy hỗ trợ các thao tác chuyển đổi ma trận ảnh
import sqlite3              # thư viện hỗ trợ thao tác với cơ sở dữ liệu
# import face_recognition
import dlib
import pyttsx3              # thư viện hỗ trợ giọng nói

ass = pyttsx3.init()        #khởi tạo một trợ lí ảo
voice = ass.getProperty('voices')# thiết lập thư viện giọng nói cho trợ lí ảo
ass.setProperty('voice', voice[1].id)# thiết lập giọng nữ cho trợ lí ảo


def speak(audio):                           #khai báo một hàm speak
    print("[ASSISTANT]  " + str(audio))     #in lên terminal thông báo
    ass.say(audio)                          #phát lên âm thanh thông báo 
    ass.runAndWait()                        

def main_getData():     #khai báo một hàm main_getData để lấy dữ liệu khuôn mặt
    speak("we're going to get face data, it takes a second")
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')# load model nhận diện khuôn mặt
    print("[INFO] loading HOG Face Detector...")
    hogFaceDetector = dlib.get_frontal_face_detector()
    
    dataBasePath = 'dataBase.db'                                            #đường dẫn tới cơ sở dữ liệu
    def insertOrUpdate(id, name): # hàm thêm mới hoặc nâng cấp cơ sở dữ liệu
        conn = sqlite3.connect(dataBasePath)#kết nối tới cơ sở dữ liệu
        # print('done1')
        query = "SELECT * FROM people WHERE ID="+str(id) # thiết lập lệnh truy cập CSDL
        # print('done2')
        cursor = conn.execute(query)    #con trỏ thực thi câu lệnh
        # print('done3')
        isRecordExist = 0               # nếu chưa có bản ghi nào
        # print('done4')
        for row in cursor:
            isRecordExist = 1           #thì sẽ ghi dữ liệu
        # print('done5')
        if(isRecordExist == 0):
            query = "INSERT INTO people(ID,Name) VALUES("+str(id)+",'"+str(name)+"')"#thêm mới dữ liệu vào CSDL với tên và id người dùng
        else: 
            query = "UPDATE people SET Name='"+str(name)+"' WHERE ID="+str(id)# nếu đã có bản thì thì thực hiện thêm mới
            
        conn.execute(query)
        conn.commit()
        conn.close()
    cap = cv2.VideoCapture(0)          # khởi tạo camera
    speak('please enter your ID:')  
    id = input()                       #từ bàn phím nhập id người dùng
    speak('please enter your name:')    
    name = input()                     #từ bàn phím nhập tên người dùng
    
    insertOrUpdate(id,name)            #thực hiện thêm mới hoặc nâng cấp CSDL

    if not os.path.exists('dataSet'):   #nếu chưa có thư mục dataSet
        os.mkdir('dataSet')             #tạo mới thư mục có tên là dataSet
            
    os.mkdir(f'dataSet/{name}')         #tạo mới thư mục mang tên người dùng trong thư mục dataSet
    roi = -1                            #biến roi để phục vụ việc lưu ảnh crop vào sát khuôn mặt
    sample = 0                          #biến đếm để lưu ảnh
    
    speak("Video Capture is now starting please stay still...")
    print("[INFO] loading HOG Face Detector...")
    
    while True: 
        ret, frame = cap.read()#đọc ảnh từ camera
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# chuyển đổi khung hình từ camera trả về thành màu đen trắng
        
        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)#nhận diện nhiều khuôn mặt trong khung hình
        faces = hogFaceDetector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
        
        # for ( x,y,w,h) in faces:# với mỗi khuôn mặt trong khung hình
            # cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)#vẽ hình vuông bo quanh khuôn mặt
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2 )
            
            sample +=1                                          #tăng biến đếm lên 1 đơn vị
            
            # roi = cv2.resize(gray[y:y+h, x:x+w], (224,224))     #ảnh roi sẽ khung hình có độ dài và rộng như hình vuông
            roi = cv2.resize(gray[y1:y2, x1:x2], (224,224)) 
            
            cv2.imwrite(f'dataSet/{name}/{id}_{sample}.jpg', roi)#lưu ảnh vào đúng tên người dùng được nhập ở trên
            print(f'dataSet/{name}/{id}_{sample}.jpg')
            
        cv2.imshow('Face Detection', frame)#hiện lên khung hình đã được nhận diện khuôn mặt
        
        if (cv2.waitKey(1) & 0xFF == 27):#chương trình đợi đến khi người dùng nhấn một phím
            break # nếu người dùng nhấm phím 'q' thì sẽ thoát chương trình lặp vô hạn
        
        if sample > 200:#nếu số ảnh đã lớn hơn 400 thì kết thúc chương trình
            break
    speak('get data is done, sir!')
    cap.release()           # tắt camera
    cv2.destroyAllWindows() # đóng tất cả các cửa sổ vừa được hiện lên
    
if __name__ == '__main__':
    main_getData()     # nếu chương trình này được gọi thì sẽ chạy hàm main_getData