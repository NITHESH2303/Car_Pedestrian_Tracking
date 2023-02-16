import cv2
from random import randrange

car_tracker = cv2.CascadeClassifier('cars.xml')
# ped_tracker = cv2.CascadeClassifier('haarcascade_fullbody.xml')

video_images = '.spyproject/img/frame'

cam = cv2.VideoCapture('rtsp://162it171:162it171@10.10.133.1:554/live')

frameFrequency = 30

total_frame = 0
id = 0
while True:

    frame_read, frame = cam.read()

    if frame_read is False:
        break
    total_frame += 1

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(grayscale_img)
    # peds = ped_tracker.detectMultiScale(grayscale_img)

    found = False

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(255), randrange(256)), 2)
        found = True

    if found:
        if total_frame % frameFrequency == 0:
            id += 1
            image_name = video_images + str(id) + '.jpg'
            cv2.imwrite(image_name, frame)
            print(image_name)

    # for (x, y, w, h) in peds:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(255), randrange(256)), 2)

    cv2.imshow('Car', frame)
    key = cv2.waitKey(1)
    #
    if key == 81 or key == 113:
        break

cam.release()

print("completed")

