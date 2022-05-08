import os                   # thư viện hỗ trợ các thao tác với file, hệ điều hành
import cv2                  # thư viện hỗ trợ xử lí hình ảnh
import numpy as np          # thư viện numpy hỗ trợ các thao tác chuyển đổi ma trận ảnh
from PIL import Image       #thư viện hỗ trợ các thao tác với chuyển đổi, đọc ảnh
# import face_recognition

import pyttsx3              #thư viện hỗ trợ giọng nói
ass = pyttsx3.init()        #khởi tạo một trợ lí ảo
voice = ass.getProperty('voices')# thiết lập thư viện giọng nói cho trợ lí ảo
ass.setProperty('voice', voice[1].id)# thiết lập giọng nữ cho trợ lí ảo


def speak(audio):                           #khai báo một hàm speak
    print("[ASSISTANT]  " + str(audio))     #in lên terminal thông báo
    ass.say(audio)                          #phát lên âm thanh thông báo 
    ass.runAndWait()  


def main_trainData():
    trainer = cv2.face.LBPHFaceRecognizer_create() # load model hỗ trợ huấn luyện dữ liệu của opencv
    
    def getImageWithId(path):#khai báo hàm lấy id, ảnh và đường dẫn ảnh
        imagePaths = []         #khai báo một mảng rỗng imagePaths
        users =[]               #khai báo một mảng rỗng users
        ids = []                #khai báo một mảng rỗng chứa id
        imgNPs = []             #khai báo một mảng rỗng chứa ảnh đã được ma trận hóa
        for user in os.listdir(path):# trong mỗi tên người dùng trong thư mục dataSet
            users.append(user)       # thêm vào mảng users tên của các người dùng
        for name in users:           # trong mỗi thư mục người dùng
            user_path = os.path.join(path, name) # nối đường dẫn theo định dạng 'dataSet/[tên người dùng]'
            for img in os.listdir(user_path):    # trong mỗi ảnh trong thư mục người dùng
                imagePaths.append(os.path.join(user_path, img))# nối chuỗi theo định dạng 'dataSet/[tên người dùng]/[id_số đếm]' và lưu vào mảng
        for img_path in imagePaths:   #với mỗi ảnh ta vừa lấy được ở trên
            print("[INFO] "+ str(img_path))   # in lên terminal
            id = img_path.split('\\')[2].split('_')[0] # tách lấy id
            id = int(id) #ép kiểu dữ liệu về integer
            print(id)         
            ids.append(id)# thêm vào mảng id
        ids = np.array(ids)# ma trận hóa mảng id
        for img in imagePaths:# trong mỗi ảnh trong mảng ảnh đã lấy được
            faceImage = Image.open(img).convert('L')#chuyển đổi định dạng sang màu trắng đen
            faceNP = np.array(faceImage, 'uint8')   # ma trận hóa ảnh với kiểu dữ liệu uint8
            imgNPs.append(faceNP)#thêm vào mảng imgNPs
            print(faceNP)
        print("[INFO] total of image is "+str(len(imagePaths)) )
        return ids, imgNPs, imagePaths# trả về giá trị 3 mảng chính gồm id, ma trận ảnh và đường dẫn ảnh
    
    path = 'dataSet'
    
    ids, imgNPs, imagePaths = getImageWithId(path)# lấy id, ma trận ảnh và đường dẫn
    print("[INFO] it takes a time to train the model, please wait me a second...")
    trainer.train(imgNPs, ids)# huấn luyện dữ liệu với id và ma trận ảnh tương ứng của từng người dùng
    print("[INFO] writing ...")
    trainer.write('training.yml')# sau khi huấn luyện dữ liệu thì  lưu vào file training.yml
    speak('training model is done')#in lên terminal thông báo
    cv2.destroyAllWindows()#đóng tất cả các cửa sổ
    
if __name__ == '__main__':
    main_trainData()# nếu chương trình được chạy thì sẽ thực thi hàm main_trainData