import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import random
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("/Users/aknur/Downloads/converted_keras/keras_model.h5", "/Users/aknur/Downloads/converted_keras/labels.txt")

offset, imgSize = 20, 300

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M", "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
SIGNS_DIR = "/Users/aknur/Desktop/CVision/project-asl/signs-imgs"

sign_images = {}
for label in labels:
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(SIGNS_DIR, f"{label}{ext}")
        if os.path.exists(path):
            img = cv2.imread(path)
            sign_images[label] = cv2.resize(img, (120, 120))
            break
target_letter = random.choice(labels)
hold_time_required = 15
hold_counter = 0

def draw_top_toolbar(img, target_letter, sign_accuracy, is_correct):
    overlay = img.copy()
    h, w, _ = img.shape

    y1, y2 = 0, 90
    cv2.rectangle(overlay, (0, y1), (w, y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    color = (0, 200, 0) if is_correct else (0, 0, 200)
    cv2.line(img, (0, y2), (w, y2), color, 3)
    cv2.putText(img, f"TARGET: {target_letter}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(img, f"| sign accuracy: {sign_accuracy}%", (300, 62),cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    cv2.putText(img, "| N: next | Q: quit", (w - 300, 62),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    predicted_label = ""
    sign_accuracy = 0
    is_correct = False

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop  = img[y1:y2, x1:x2]
        if imgCrop.size != 0:
            aspectRatio = h / w
            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                predicted_label = labels[index]
                target_index = labels.index(target_letter)
                sign_accuracy = int(prediction[target_index] * 100)

                is_correct = predicted_label == target_letter
                if is_correct:
                    hold_counter += 1
                else:
                    hold_counter = 0

                if hold_counter >= hold_time_required:
                    target_letter = random.choice(labels)
                    hold_counter = 0
                color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.rectangle(imgOutput,(x - offset, y - offset), (x + w + offset, y + h + offset),color, 3)

                label_text = f"{predicted_label} ({sign_accuracy}%)"

                cv2.rectangle(imgOutput, (x - offset, y - offset - 40), (x - offset + 200, y - offset), color, cv2.FILLED)
                cv2.putText(imgOutput, label_text,(x - offset + 5, y - offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                bar_x1 = x - offset
                bar_x2 = x + w + offset
                bar_y  = y + h + offset + 12
                progress = int((hold_counter / hold_time_required) * (bar_x2 - bar_x1))
                cv2.rectangle(imgOutput, (bar_x1, bar_y), (bar_x2, bar_y + 8), (60, 60, 60), -1)
                if progress > 0:
                    cv2.rectangle(imgOutput, (bar_x1, bar_y), (bar_x1 + progress, bar_y + 8), (0, 255, 0), -1)

            except Exception as e:
                print("error:", e)
    draw_top_toolbar(imgOutput, target_letter, sign_accuracy, is_correct)

    if target_letter in sign_images:
        hint = sign_images[target_letter]
        imgOutput[100:220, 20:140] = hint
    key = cv2.waitKey(1) & 0xFF

    if key == ord('n'):
        target_letter = random.choice(labels)
        hold_counter = 0
    if key == ord('q'):
        break
    cv2.imshow("ASL Learning Game", imgOutput)
cap.release()
cv2.destroyAllWindows()