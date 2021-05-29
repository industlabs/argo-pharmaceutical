#!/usr/bin/python
import os
import time
# from vimba import *
import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from pyModbusTCP.client import ModbusClient

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

part1_casc = cv2.CascadeClassifier('main_casc/cascade1.xml')
part2_casc = cv2.CascadeClassifier('main_casc/cascade2.xml')
part3_casc = cv2.CascadeClassifier('main_casc/cascade3.xml')
part4_casc = cv2.CascadeClassifier('main_casc/cascade4.xml')
part5_casc = cv2.CascadeClassifier('main_casc/cascade5.xml')
part6_casc = cv2.CascadeClassifier('main_casc/cascade6.xml')
part7_casc = cv2.CascadeClassifier('main_casc/cascade7.xml')
part8_casc = cv2.CascadeClassifier('main_casc/cascade8.xml')
part9_casc = cv2.CascadeClassifier('main_casc/cascade9.xml')
part10_casc = cv2.CascadeClassifier('main_casc/cascade10.xml')
part11_casc = cv2.CascadeClassifier('main_casc/cascade11.xml')
partpik_casc = cv2.CascadeClassifier('main_casc/cascadepik.xml')

part1_1_casc = cv2.CascadeClassifier('sub_casc/cascade1.1.xml')
part1_2_casc = cv2.CascadeClassifier('sub_casc/cascade1.2.xml')
part1_3_casc = cv2.CascadeClassifier('sub_casc/cascade1.3.xml')
part2_1_casc = cv2.CascadeClassifier('sub_casc/cascade2.1.xml')
part2_2_casc = cv2.CascadeClassifier('sub_casc/cascade2.2.xml')
part2_3_casc = cv2.CascadeClassifier('sub_casc/cascade2.3.xml')
part3_1_casc = cv2.CascadeClassifier('sub_casc/cascade1.1.xml')
part3_2_casc = cv2.CascadeClassifier('sub_casc/cascade1.2.xml')
part3_3_casc = cv2.CascadeClassifier('sub_casc/cascade1.3.xml')
part4_1_casc = cv2.CascadeClassifier('sub_casc/cascade4.1.xml')
part4_2_casc = cv2.CascadeClassifier('sub_casc/cascade4.2.xml')
part5_1_casc = cv2.CascadeClassifier('sub_casc/cascade5.1.xml')
part5_2_casc = cv2.CascadeClassifier('sub_casc/cascade5.2.xml')
part6_1_casc = cv2.CascadeClassifier('sub_casc/cascade6.1.xml')
part6_2_casc = cv2.CascadeClassifier('sub_casc/cascade6.2.xml')
part6_3_casc = cv2.CascadeClassifier('sub_casc/cascade6.3.xml')
part6_4_casc = cv2.CascadeClassifier('sub_casc/cascade6.4.xml')
part6_up_casc = cv2.CascadeClassifier('sub_casc/cascade6.up.xml')
part6_dowm_casc = cv2.CascadeClassifier('sub_casc/cascade6.down.xml')
part7_1_casc = cv2.CascadeClassifier('sub_casc/cascade7.1.xml')
part7_2_casc = cv2.CascadeClassifier('sub_casc/cascade7.2.xml')
part8_1_casc = cv2.CascadeClassifier('sub_casc/cascade8.1.xml')
part8_2_casc = cv2.CascadeClassifier('sub_casc/cascade8.2.xml')
part8_3_casc = cv2.CascadeClassifier('sub_casc/cascade8.3.xml')
part8_4_casc = cv2.CascadeClassifier('sub_casc/cascade8.4.xml')
part9_1_casc = cv2.CascadeClassifier('sub_casc/cascade9.1.xml')
part9_2_casc = cv2.CascadeClassifier('sub_casc/cascade9.2.xml')
part9_3_casc = cv2.CascadeClassifier('sub_casc/cascade9.3.xml')
part9_4_casc = cv2.CascadeClassifier('sub_casc/cascade9.4.xml')
part10_1_casc = cv2.CascadeClassifier('sub_casc/cascade10.1.xml')
part10_2_casc = cv2.CascadeClassifier('sub_casc/cascade10.2.xml')
part10_3_casc = cv2.CascadeClassifier('sub_casc/cascade10.3.xml')
part10_4_casc = cv2.CascadeClassifier('sub_casc/cascade10.4.xml')
part11_1_casc = cv2.CascadeClassifier('sub_casc/cascade11.1.xml')
part11_2_casc = cv2.CascadeClassifier('sub_casc/cascade11.2.xml')
part11_3_casc = cv2.CascadeClassifier('sub_casc/cascade11.3.xml')
part11_4_casc = cv2.CascadeClassifier('sub_casc/cascade11.4.xml')

# Load keras model
main_model = tensorflow.keras.models.load_model('main_keras_models/keras_model_11Parts.h5')
three_parts_model = tensorflow.keras.models.load_model('main_keras_models/keras_model_3Parts.h5')
pikpoint_model = tensorflow.keras.models.load_model('main_keras_models/keras_model_pikpoint.h5')

model1 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_1.h5')
model2 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_2.h5')
model3 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_1.h5')
model4 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_4.h5')
model5 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_5.h5')
model6 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_6.h5')
model7 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_7.h5')
model8 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_8.h5')
model9 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_9.h5')
model10 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_10.h5')
model11 = tensorflow.keras.models.load_model('sub_keras_models/keras_model_11.h5')
model11_2 = tensorflow.keras.models.load_model('sub_keras_models/keras_model11.2.h5')

nextreg = 20


# cv2.namedWindow('Indust Labs Object Detection Program', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

def resize(img, scale):
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def processImage(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data


def analyze1(cropped):
    orientation = 0
    pick_point = [0, 0, 0, 0]

    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for j in prediction[0]:
        count += 1
        if j > 0.6:
            orientation = count

    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(0, 10))
    for (x, y, w, h) in detectpik:
        recropped = cropped[y:y + h, x:x + w]
        datapick = processImage(recropped)
        prediction = pikpoint_model.predict(datapick)
        if prediction[0][0] > 0.8:
            pick_point = [x, y, w, h]

    return orientation, pick_point


def analyze2(cropped):
    orientation = 0
    pick_point = [0, 0, 0, 0]

    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for j in prediction[0]:
        count += 1
        if j > 0.6:
            orientation = count

    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(0, 10))
    for (x, y, w, h) in detectpik:
        recropped = cropped[y:y + h, x:x + w]
        datapick = processImage(recropped)
        prediction = pikpoint_model.predict(datapick)
        if prediction[0][0] > 0.8:
            pick_point = [x, y, w, h]

    return orientation, pick_point


def analyze3(cropped):
    orientation = 0
    pick_point = [0, 0, 0, 0]

    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count

    detectpik = partpik_casc.detectMultiScale(cropped, 1.1, 1, minSize=(0, 10))
    for (x, y, w, h) in detectpik:
        recropped = cropped[y:y + h, x:x + w]
        datapick = processImage(recropped)
        prediction = pikpoint_model.predict(datapick)
        if prediction[0][0] > 0.8:
            pick_point = [x, y, w, h]

    return orientation, pick_point


def analyze4(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze5(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze6(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze7(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze8(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze9(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze10(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            orientation = count
    return orientation


def analyze11(cropped):
    orientation = 0
    data = processImage(cropped)
    prediction = model1.predict(data)
    count = 0
    for i in prediction[0]:
        count += 1
        if i > 0.6:
            if i == 2 or i == 4:
                prediction2 = model11_2.predict(data)
                if prediction2[0][0] > 0.5:
                    orientation = 2
                elif prediction2[0][1] > 0.5:
                    orientation = 4
            else:
                orientation = count
    return orientation


def sendData(partNo, pickpoinX, pickpointY, orientationNo, count):
    global nextreg
    client.write_multiple_registers(nextreg, [partNo, pickpoinX, pickpointY, orientationNo])
    nextreg += 10
    client.write_single_register(partNo, count)


def deepCheck(imgdata, partNo):
    prediction = three_parts_model.predict(imgdata)

    if prediction[0][partNo - 1] > 0.9:
        return True
    else:
        return False


def verifyPart(img, partNo):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = main_model.predict(data)
    if partNo > 3 and prediction[0][partNo - 1] > 0.9:
        return True
    elif partNo <= 3 and prediction[0][0] > 0.9:
        isTrue = deepCheck(data, partNo)
        return isTrue
    else:
        return False


def findObjects(input_path, output_path):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0
    count8 = 0
    count9 = 0
    count10 = 0
    count11 = 0
    found = False

    img = cv2.imread(input_path)
    # alpha = 0.8
    # beta = 61
    # # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
    # img = cv2.addWeighted(img, alpha, img, 0, beta)
    # alpha2 = 2
    # beta2 = -100
    # img = cv2.addWeighted(img, alpha2, img, 0, beta2)
    # blur = cv2.GaussianBlur(lowBright, (5,5), 0)

    # with Vimba.get_instance() as vimba:
    #     cams = vimba.get_all_cameras()
    #     with cams[0] as cam:
    #         while True:
    #             frame = cam.get_frame()
    #
    #             frame.convert_pixel_format(PixelFormat.Bgr8)
    # img = frame.as_opencv_image()

    # img = resize(img, 60)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detect1 = part1_casc.detectMultiScale(gray, 1.01, 2, minSize=(250, 250))
    detect2 = part2_casc.detectMultiScale(gray, 1.01, 2, minSize=(300, 300))
    detect3 = part3_casc.detectMultiScale(gray, 1.01, 2, minSize=(300, 300))
    detect4 = part4_casc.detectMultiScale(gray, 1.01, 5, minSize=(400, 400), maxSize=(650, 650))
    detect5 = part5_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect6 = part6_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect7 = part7_casc.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))
    detect8 = part8_casc.detectMultiScale(gray, 1.1, 5, minSize=(200, 200))
    detect9 = part9_casc.detectMultiScale(gray, 1.1, 5, minSize=(500, 500))
    detect10 = part10_casc.detectMultiScale(gray, 1.1, 5, minSize=(90, 90))
    detect11 = part11_casc.detectMultiScale(gray, 1.01, 3, minSize=(90, 90))

    for (x, y, w, h) in detect1:
        count1 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        # "35-1205-03" 99-402019000
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 1):
            continue
        orientation, pick_point = analyze1(cropped)
        pickpoint = (pick_point[0] + int(pick_point[2] / 2), pick_point[1] + int(pick_point[3] / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 1" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 1", end=",")
        sendData(1, pickpoint[0], pickpoint[1], orientation, count1)
        found = True

    for (x, y, w, h) in detect2:
        count2 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        # "35-1205-03"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 2):
            continue
        orientation, pick_point = analyze2(cropped)
        pickpoint = (pick_point[0] + int(pick_point[2] / 2), pick_point[1] + int(pick_point[3] / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 2" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 2", end=",")
        sendData(2, pickpoint[0], pickpoint[1], orientation, count2)
        found = True

    for (x, y, w, h) in detect3:
        count3 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        # "35-1205-03"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 3):
            continue
        orientation, pick_point = analyze3(cropped)
        pickpoint = (pick_point[0] + int(pick_point[2] / 2), pick_point[1] + int(pick_point[3] / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        # cv2.rectangle(img, (x+pick_point[0], y+pick_point[1]), ( x+pick_point[0]+pick_point[2], y+pick_point[1]+pick_point[3]), (0, 255, 0), 5)
        cv2.putText(img, "Part 3" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 3", end=",")
        sendData(3, pickpoint[0], pickpoint[1], orientation, count3)
        found = True

        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)

    for (x, y, w, h) in detect4:
        count4 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 4):
            continue
        orientation = analyze4(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 4" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        # print("Part 4", end=",")
        sendData(4, pickpoint[0], pickpoint[1], orientation, count4)
        found = True

    for (x, y, w, h) in detect5:
        count5 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 5):
            continue
        orientation = analyze5(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 5" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 5", end=",")
        sendData(5, pickpoint[0], pickpoint[1], orientation, count5)
        found = True

    for (x, y, w, h) in detect6:
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 6):
            continue
        count6 += 1
        orientation = analyze6(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 6" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 6", end=",")
        sendData(6, pickpoint[0], pickpoint[1], orientation, count6)
        found = True

    for (x, y, w, h) in detect7:
        count7 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 7):
            continue
        orientation = analyze7(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 7" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 7", end=",")
        sendData(7, pickpoint[0], pickpoint[1], orientation, count7)
        found = True

    for (x, y, w, h) in detect8:
        count8 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 8):
            continue
        orientation = analyze8(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 8" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 8", end=",")
        sendData(8, pickpoint[0], pickpoint[1], orientation, count8)
        found = True

    for (x, y, w, h) in detect9:
        count9 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 9):
            continue
        orientation = analyze9(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 9" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 9", end=",")
        sendData(9, pickpoint[0], pickpoint[1], orientation, count9)
        found = True

    for (x, y, w, h) in detect10:
        count10 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 10):
            continue
        orientation = analyze10(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 10" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, pickpoint, 5, (255, 0, 0), -1)
        # print("Part 10", end=",")
        sendData(10, pickpoint[0], pickpoint[1], orientation, count10)
        found = True

    for (x, y, w, h) in detect11:
        count11 += 1
        sizeText = "(" + str(w) + "," + str(h) + ")"
        cropped = img[y:y + h, x:x + w]
        if not verifyPart(cropped, 11):
            continue
        orientation = analyze11(cropped)
        pickpoint = (x + int(w / 2), y + int(h / 2))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(img, "Part 11" + sizeText + "ori:" + str(orientation) + str(pickpoint), (x, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        cv2.circle(img, (x + int(w / 2), y + int(h / 2)), 5, (255, 0, 0), -1)
        # print("Part 11", end=",")
        sendData(11, pickpoint[0], pickpoint[1], orientation, count11)
        found = True

    # print()
    cv2.putText(img, "count: " + str(count1), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)

    # cv2.imshow("Indust Labs Object Detection Program", img)
    cv2.imwrite(output_path, img)
    cv2.waitKey(1)
    # if cv2.waitKey(5000) & 0xFF == ord('\r'):
    #     pass
    cv2.destroyAllWindows()
    if found and count6 > 15:
        print("count6:", count6)
        print(input_path)
        print(output_path)
    return found


# if cv2.waitKey(1) & 0xFF == ord('\r'):
#     break

# Delete files older than 30 days
def deleteOldFiles(path):
    try:
        timenow = time.time()
        deadline = timenow - (2592000)

        for filename in os.listdir(path):
            output_file_path = path + filename
            t = os.path.getctime(output_file_path)
            if t < deadline:
                try:
                    os.remove(output_file_path)
                except OSError as e:  # if failed, report it back to the user ##
                    print("Error: %s - %s." % (e.filename, e.strerror))

    except Exception as e:
        print(e)


# Driver Code
if __name__ == '__main__':
    # C:\Users\Administrator\Desktop\TestStage4\input
    path = "C:/Users/Administrator/Desktop/TestStage4/input/"
    output = "C:/Users/Administrator/Desktop/TestStage4/output/"

    deleteOldFiles(output)

    try:
        while True:
            client = ModbusClient(host="localhost", port=502)
            client.open()
            if client.is_open():
                client.write_single_register(0, 0)
                break
    except:
        print("Unable to initialize the modbus client or open the modbus connection.")
        print("Press enter to exit.")
        exitInput = input()
        exit(1)

    i = 1

    while True:
        for filename in os.listdir(path):

            while True:
                # fetch first register
                reg1 = client.read_holding_registers(0)
                # check data bank reset
                if not client.is_open():
                    print("Cannot read from modbus server", i)
                    time.sleep(2)
                    continue
                elif reg1[0] == 0:
                    break
            # print(i, " ", time.time())
            # global nextreg
            nextreg = 20
            output_file = str(i) + ".jpg"
            input_path = path + filename
            output_path = output + output_file

            try:
                found = findObjects(input_path, output_path)
                if found:
                    statuscheck = client.write_single_register(0, i)
                # Try to delete the file
                try:
                    os.remove(input_path)
                except OSError as e:  # if failed, report it back to the user ##
                    print("Error: %s - %s." % (e.filename, e.strerror))
            except Exception as e:
                print("error in finding objects", input_path)
                print(e)

            # findObjects(input_path, output_path)
            # statuscheck = client.write_single_register(0,i)
            # ## Try to delete the file ##
            # try:
            #     os.remove(input_path)
            # except OSError as e:  ## if failed, report it back to the user ##
            #     print("Error: %s - %s." % (e.filename, e.strerror))

            # time.sleep(1)
            # print(i, "+ ", time.time())
            i += 1

    client.close()
    cv2.destroyAllWindows()
    exit(0)
