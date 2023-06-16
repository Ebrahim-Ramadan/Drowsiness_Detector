import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from keras.models import load_model
import numpy as np
import pygame
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound("sounds/alert.wav")

face = cv2.CascadeClassifier(
    'haarcascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(
    'haarcascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(
    'haarcascade files/haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]


class DrowsinessDetector:
    def __init__(self, master):
        self.eyes_open = True
        self.sound_played = False
        self.master = master
        self.master.title("Drowsiness Detector")
        self.master.resizable(False, False)
        self.sound = mixer.Sound('sounds/alert.wav')
        self.video_capture = cv2.VideoCapture(0)
        self.image = None
        self.is_running = False
        self.canvas_image = None
        self.canvas = tk.Canvas(self.master, width=640, height=480, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.start_button = ctk.CTkButton(self.master, text="Start", command=lambda: (
            self.start_detection(), self.detect_drowsiness()))
        self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')

        self.stop_button = ctk.CTkButton(
            self.master, text="Stop", command=self.stop_detection)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

        self.status_label = tk.Label(self.master, text="Status: Idle")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        self.image_label = tk.Label(self.master, image=None)
        self.image_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        self.update_canvas()

    def start_detection(self):
        self.video_capture = cv2.VideoCapture(0)
        self.is_running = True
        self.update_canvas()
        self.status_label.config(text="Status: Running")

    def stop_detection(self):
        self.is_running = False
        if self.video_capture is not None:
            self.video_capture.release()
        self.canvas.delete("all")
        self.status_label.config(text="Status: Idle")

    def update_canvas(self):
        frame = None
        count = 0
        score = 0
        self.closed_frames_count = 0  # Counter for consecutive closed eye frames
        self.wait_frames_count = 0  # Counter for frames to wait after closed eyes

        thicc = 2
        rpred = [99]
        lpred = [99]

        if self.is_running == True:
            ret, frame = self.video_capture.read()
            if ret:
                height, width = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(
                    gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
                left_eye = leye.detectMultiScale(gray)
                right_eye = reye.detectMultiScale(gray)

                cv2.rectangle(frame, (0, height - 50), (200, height),
                              (0, 0, 0), thickness=cv2.FILLED)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (100, 100, 100), 1)

            for (x, y, w, h) in right_eye:
                r_eye = frame[y:y + h, x:x + w]
                count = count + 1
                r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye, (24, 24))
                r_eye = r_eye / 255
                r_eye = r_eye.reshape(24, 24, -1)
                r_eye = np.expand_dims(r_eye, axis=0)

                predict_x = model.predict(r_eye)
                rpred = np.argmax(predict_x, axis=1)

                if (rpred[0] == 1):
                    lbl = 'Open'
                if (rpred[0] == 0):
                    lbl = 'Closed'
                break

            for (x, y, w, h) in left_eye:
                l_eye = frame[y:y + h, x:x + w]
                count = count + 1
                l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
                l_eye = cv2.resize(l_eye, (24, 24))
                l_eye = l_eye / 255
                l_eye = l_eye.reshape(24, 24, -1)
                l_eye = np.expand_dims(l_eye, axis=0)
                predict_x = model.predict(l_eye)
                lpred = np.argmax(predict_x, axis=1)

                if (lpred[0] == 1):
                    lbl = 'Open'
                if (lpred[0] == 0):
                    lbl = 'Closed'

                if (rpred[0] == 0 and lpred[0] == 0):
                    score = score + 1
                    self.closed_frames_count += 1
                    cv2.putText(frame, "Closed", (10, height - 20),
                                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                    self.sound.play()  #
                else:
                    score = score - 1
                    self.closed_frames_count = 0
                    cv2.putText(frame, "Open", (10, height - 20),
                                font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                if (score < 0):
                    score = 0
                cv2.putText(frame, 'Score:' + str(score), (100, height - 20),
                            font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                if self.closed_frames_count >= 40:
                    self.wait_frames_count += 1
                    if self.wait_frames_count >= 60:
                        # person's eyes are closed for consecutive frames, trigger the alarm
                        # cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                        self.sound.play()
                        if (thicc < 16):
                            thicc = thicc + 2
                        else:
                            thicc = thicc - 2
                            if (thicc < 2):
                                thicc = 2
                        cv2.rectangle(frame, (0, 0), (width, height),
                                      (0, 0, 255), thicc)
                else:
                    self.wait_frames_count = 0
                    # self.sound.stop()

            if cv2.waitKey(33) & 0xFF == ord('q'):
                self.stop_detection()

            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.canvas_image = ImageTk.PhotoImage(
                image=Image.fromarray(self.image))
            self.canvas.create_image(
                0, 0, anchor='nw', image=self.canvas_image)
            self.master.after(20, self.update_canvas)

    def detect_drowsiness(self):
        if self.image is not None:
            img = cv2.resize(self.image, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = img/255.0
            prediction = model.predict(img)
            if prediction > 0.5:
                self.status_label.config(text="Status: Alert")
                # Play the sound only if it hasn't already been played and the eyes were previously open
                if not self.sound_played and self.eyes_open:
                    self.eyes_open = False
                    self.sound.play()
                    self.sound_played = True  # Set the sound_played flag to True after playing the sound
            else:
                self.status_label.config(text="Status: Awake")
                self.sound_played = False
                self.eyes_open = True  # Set eyes_open to True when the eyes are open
        else:
            messagebox.showerror("Error", "No frame available for detection.")


root = tk.Tk()
app = DrowsinessDetector(root)
root.mainloop()
