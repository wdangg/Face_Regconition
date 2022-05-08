from getData import main_getData
from trainData import main_trainData
from faceRecognition import main_faceRecognition
import numpy as np          # thư viện numpy hỗ trợ các thao tác chuyển đổi ma trận ảnh
import cv2                  # thư viện hỗ trợ xử lí hình ảnh
import os                   # thư viện hỗ trợ các thao tác với file, hệ điều hành
import sqlite3              # thư viện hỗ trợ thao tác với cơ sở dữ liệu
from PIL import Image       #thư viện hỗ trợ các thao tác với chuyển đổi, đọc ảnh
# import face_recognition
import pyttsx3                           #thư viện hỗ trợ giọng nói
ass = pyttsx3.init()                     #khởi tạo một trợ lí ảo
voice = ass.getProperty('voices')        # thiết lập thư viện giọng nói cho trợ lí ảo
ass.setProperty('voice', voice[1].id)    # thiết lập giọng nữ cho trợ lí ảo
volume = ass.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
ass.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

def speak(audio):                           #khai báo một hàm speak
    print("[ASSISTANT]  " + str(audio))     #in lên terminal thông báo
    ass.say(audio)                          #phát lên âm thanh thông báo 
    ass.runAndWait() 
def announcement():
    print("can i help you?")
    print("0. quit")
    print("1. get face data")
    print("2. train data")
    print("3. start recognize")
    smt=ord(input('please enter your command: '))# gán biến smt bằng giá trị trả về khi nhấn phím đó
    return smt# trả về giá trị biến smt

if __name__ == '__main__':
    speak('hello sir, How can i help you?')
    while True:
        smtAction = announcement()# đọc và ghi phím người dùng nhấn
        if smtAction == ord('0'): 
            speak('you wanna quit?')
            break # nếu người dùng nhấn '0' thì thoát chương trình
        elif smtAction == ord('1'):
            main_getData()# nếu người dùng nhấn phím '1' thì thực hiện lấy dữ liệu
        elif smtAction == ord('2'):
            speak('you wanna train data')
            main_trainData()#nếu ngưười dùng nhấn phím '2' thì thực hiện huấn luyện dữ liệu
        elif smtAction == ord('3'):
            main_faceRecognition()# nếu người dùng nhấn '3' thì thực hiện nhận diện khuôn mặt
            break 
        else:
            pass
    speak("all done! Have a nice day, sir!")
   