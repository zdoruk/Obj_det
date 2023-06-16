
import cv2
import json
from datetime import datetime
from picamera import PiCamera
from pyzbar import pyzbar
from time import sleep
import pyrebase
import os


# Firebase Bağlantısı İçin Gereken Bilgi Dosyası
firebaseConfig = {
    'apiKey': "-",
    'authDomain': "-",
    'databaseURL': "-",
    'projectId': "-",
    'storageBucket': "-",
    'messagingSenderId': "-",
    'appId': "-",
    'measurementId': "-"
}

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

# Borkodlar için Ürün Veri Tabanı Örneği
karsilastirma = {
        "0012345678912": "fatcow milk 1 lt",
        "0512345000107": "bonjoar cheddar cheese 400g",
        "0020357122682": "miliko dark chocolate 80g",
        "0725272730706": "bonjoar labneh 500g",
        "0076950450479": "dames fruit juice 200 ml"
}


# thres = 0.45 # Nesnenin tespit edilmesi için eşik değeri
# Görüntü Üzerindeki Nesnelerin Tanımlaması İçin Kullanılacak Dosyaların Konumları Ve Ayarları
classNames = []
classFile = "/home/zeynel/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/zeynel/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/zeynel/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId-1].upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo

# Tanımlanan Barkod Numaralarının Hangi Eşyaya Ait Olduğunu Tanımlayan Fonksiyon
def karsiliklari_bul(degerler, karsilastirma):
    esyalar = []
    for deger in degerler:
        if deger in karsilastirma:
            esyalar.append(karsilastirma[deger])
            print(f"{deger}: {karsilastirma[deger]}")
    
    return esyalar

# 10. Karedeki Görüntü Üzerinde Barkod Tanımlaması Yapılan Fonksiyon
def decode(image):
    # Barkodları çöz
    decoded_objects = pyzbar.decode(image)
    barcodes = [obj.data.decode('utf-8') for obj in decoded_objects]
    
    for obj in decoded_objects:
        # Barkod tipini ve verisini yazdır
        print("Tespit edilen barkod:")
        print("Veri:", obj.data)
        print()
    
    return barcodes

# Barkodların Tanımlaması Yapılmadan Önce 10 Karelik Bir Görütü Alındığı Fonksiyon
def capture_camera(num_frames=10):
    # Kamera bağlantısını başlat
    cap = cv2.VideoCapture(0)
    barcodes = []

    for _ in range(num_frames):
        # Kameradan bir frame yakala
        ret, frame = cap.read()

        # Frame'i göster
        cv2.imshow("Kamera", frame)

        # Barkodları tara
        if _ == num_frames - 1:
            barcodes = decode(frame)

    # Kamerayı kapat
    cap.release()
    cv2.destroyAllWindows()

    return barcodes

# İlerde Kullanılacak Bazı Değişkenler
dolapdaki_dakika = 0
icerik_listesi = []
eski_icerik = []
while True:
#   Kullanıcının ID Numarasının Girileceği Yer
    userID="-"
#   Gönderilecek Verilerin Yazılacağı JSON Dosyası Oluştutulması Ve Kontrol Edilmesi
    json_name = userID +"-j.json"
    
    if os.path.exists(json_name):
        os.remove(json_name)
        print(f"{json_name} başarıyla silindi.")
    else:
        print(f"{json_name} adında bir dosya bulunamadı.")
    
    
    
    

#   Kameranın Açılıp 7 Kare Boyunca Görüntülerin Tanımlanması
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
        
    x = 0
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, 0.45, 0.2)
        
        #print(objectInfo)
        print("Tespit Edilen Nesneler:", objectInfo)
        cv2.imshow("Output", img)
        cv2.imshow("Output", img)
        cv2.waitKey(1)
            
        x = x + 1
        if x == 7:
            x = 0
                
#       7. Karedeki Tanımlanan Nesne İsimlerini Değişkene Kaydetme
                
            eski_icerik=icerik_listesi
            icerik_listesi = []
                
            for item in objectInfo:
                icerik = item[1]
                icerik_listesi.append(icerik)
                
            image_name = userID + "-r.jpg"
            cv2.imwrite(image_name , img)
            print(image_name + " kaydedildi")
            storage.child(image_name).put(image_name)
            print("Resim gönderildi")
#           Sonsuz Döngüden Çıkılır
            break
    cap.release()
    cv2.destroyAllWindows()
    
    camera = PiCamera()
    image_name2 = userID + "-p.jpg"
    camera.capture(image_name2)
    print(image_name2 + " kaydedildi")
    storage.child(image_name2).put(image_name2)
    print("Resim2 gönderildi")
    camera.close()
    
#   İçerikleerin Listesi Bir Önceki Liste ile Karşılaştırılır Ve Değişi Yoksa Süre Artıyor
    iceriks = len(icerik_listesi)
    eskis = len(eski_icerik)
    fark = [ab for ab in icerik_listesi if ab not in eski_icerik]
    if fark==[]:
        dolapdaki_dakika = dolapdaki_dakika + 1
        
    else:
        dolapdaki_dakika=0
        
        
    cap.release()
    cv2.destroyAllWindows()
    
    detected_barcodes = capture_camera(10)
    print("Okunan barkodlar:", detected_barcodes)
    
    esyalar = karsiliklari_bul(detected_barcodes, karsilastirma)
    print("Eşyalar:", esyalar)
     
    cap.release()
    cv2.destroyAllWindows()
    
    
    now = datetime.now()
    dtd = now.strftime("%d")
    dta = now.strftime("%m")
    dty = now.strftime("%Y")
    dth = now.strftime("%H")
    dtm = now.strftime("%M")
    dts = now.strftime("%S")
    
    son_içerik_listesi = icerik_listesi + esyalar
    
    json_data = {
                    "food": son_içerik_listesi,
                    "date": {
                        "year": dty,
                        "month": dta,
                        "day": dtd,
                        "hour": dth,
                        "minute": dtm,
                        "second": dts
                    },
                    "changeFood":fark,
                    "foodChangeTimeMinute": dolapdaki_dakika
                    }
    with open(json_name, "w") as json_file:
                    json.dump(json_data, json_file)
    storage.child(json_name).put(json_name)
    print("JSON dosyası gönderildi")
    
#   İşlemin Tam Olarak 60 Saniye Sürmesi İçin Ayarlanmış Zaman Atlaması
    sleep(39)
    
    
