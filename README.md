# Drowsiness Detector Python Project Documentation

Welcome to the documentation for the **Drowsiness Detector** Python project. This project is designed to detect and alert individuals when they exhibit signs of drowsiness while driving or operating machinery. The Drowsiness Detector utilizes computer vision techniques to monitor a person's eyes and determine if they are in a drowsy state.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Drowsy driving is a serious safety concern, as it can lead to accidents due to delayed reaction times and decreased attention. The Drowsiness Detector project aims to address this issue by providing a tool that can monitor a person's eyes using a webcam and raise alerts if signs of drowsiness are detected. (very accurate regarding how closed the eyes can be)

## Features

- Real-time monitoring of a person's eyes using a webcam.
- Detection of common signs of drowsiness, such as slow eye movements and frequent blinking.
- Audible and visual alerts when drowsiness is detected.
- Adjustable sensitivity settings to accommodate different individuals.
- Lightweight and easy to set up.

## Installation

1. Clone the repository:
   ```git clone https://github.com/Ebrahim-Ramadan/Drowsiness_Detector.git```
You can configure the Drowsiness Detector by modifying the config.json file. This file allows you to adjust the following parameters:

## Configuration

You can configure the Drowsiness Detector by modifying the config.json file. This file allows you to adjust the following parameters:

- eye_aspect_ratio_threshold: Threshold for detecting closed eyes.
- ear_consecutive_frames: Number of consecutive frames for which the eye aspect ratio can be below the threshold before triggering an alert.
- alarm_sound_path: Path to the sound file that will be played when drowsiness is detected.
  
## Contributing

Contributions are welcome! If you find any issues or want to enhance the project, feel free to open pull requests. Please ensure that you follow the project's coding standards and best practices.

- Fork the repository.
- Create a new branch for your feature or bug fix: git checkout -b feature-name.
- Make your changes and commit them with descriptive commit messages.
- Push your changes to your fork: git push origin feature-name.
- Open a pull request explaining your changes and why they should be merged.


Thank you for using the Drowsiness Detector project! If you have any questions, concerns, or suggestions, please contact us at ramadanebrahim791@gmail.com. Your feedback is highly appreciated.
