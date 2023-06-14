import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
# Set up paths to your training and testing data folders
train_dir = 'archive (1)//dataset_new//train'
test_dir = 'archive (1)//dataset_new//test'

# Set up the image size and batch size for training
img_width, img_height = 224, 224
batch_size = 128

# Set up the number of epochs you want to train for
num_epochs = 1
# Set up the paths to your model file and weights file
model_file = 'cnnCat.h5'


# Define your model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile your model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Set up data augmentation for your training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

# Set up data augmentation for your testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='binary')

# Train your model
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=num_epochs, validation_data=test_generator, validation_steps=len(test_generator))

# Save your model and weights
model.save(model_file)


# Load your model and weights
model = load_model(model_file)

face_cascade_file = 'haarcascade_frontalface_alt.xml'
leye_cascade_file = 'haarcascade_lefteye_2splits.xml'
reye_cascade_file = 'haarcascade_righteye_2splits.xml'


if not os.path.exists(face_cascade_file):
    raise FileNotFoundError(f"{face_cascade_file} not found.")
if not os.path.exists(leye_cascade_file):
    raise FileNotFoundError(f"{leye_cascade_file} not found.")
if not os.path.exists(reye_cascade_file):
    raise FileNotFoundError(f"{reye_cascade_file} not found.")
if not os.path.exists(model_file):
    raise FileNotFoundError(f"{model_file} not found.")

face = cv2.CascadeClassifier(face_cascade_file)
leye = cv2.CascadeClassifier(leye_cascade_file)
reye = cv2.CascadeClassifier(reye_cascade_file)



class DrowsinessDetector:
    def __init__(self, master):
        self.master = master
        self.master.title("Drowsiness Detector")
        self.master.resizable(False, False)
        mixer.init()
        self.sound = mixer.Sound('alarm.wav')
        self.video_capture = None
        self.image = None
        self.is_running = False
        self.canvas_image = None
        self.sound_played = False
        self.canvas = tk.Canvas(self.master, width=640, height=480, bg='black')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        
        self.start_button = tk.Button(self.master, text="Start", command=self.start_detection)
        self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky='w')
        
        self.stop_button = tk.Button(self.master, text="Stop", command=self.stop_detection)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')
        
        self.status_label = tk.Label(self.master, text="Status: Idle")
        self.status_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)
        
        self.detect_button = tk.Button(self.master, text="Detect", command=self.detect_drowsiness)
        self.detect_button.grid(row=3, column=0, padx=10, pady=10, sticky='w')
        
        self.image_label = tk.Label(self.master, image=None)
        self.image_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        
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
        if self.is_running==True:
            ret, frame = self.video_capture.read()
            if ret:    
                height, width = frame.shape[:2] 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = frame[y:y+h, x:x+w]
                    left_eye = leye.detectMultiScale(roi_gray, minNeighbors=5, scaleFactor=1.1, minSize=(20, 20))
                    right_eye = reye.detectMultiScale(roi_gray, minNeighbors=5, scaleFactor=1.1, minSize=(20, 20))
                    for (ex, ey, ew, eh) in left_eye:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                    for (ex, ey, ew, eh) in right_eye:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (0, 0, 255), 2)
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.canvas_image = ImageTk.PhotoImage(image=Image.fromarray(self.image))
                self.canvas.create_image(0, 0, anchor='nw', image=self.canvas_image)
                self.detect_drowsiness()
                self.master.after(20, self.update_canvas)
            
    def detect_drowsiness(self):
        if self.image is not None:
            img = cv2.resize(self.image, (224, 224))
            img = np.expand_dims(img, axis=0)
            img = img/255.0
            prediction = model.predict(img)
            if prediction > 0.5:
                self.status_label.config(text="Status: Alert")
                if not self.sound_played:  # play the sound only if it hasn't already been played
                    self.sound.play()
                    self.sound_played = True
            else:
                self.status_label.config(text="Status: Awake")
                self.sound_played = False  # reset the boolean variable if the user's eyes are open
        else:
            messagebox.showerror("Error", "No frame available for detection.")
if __name__ == '__main__':
    root = tk.Tk()
    app = DrowsinessDetector(root)
    root.mainloop()
 