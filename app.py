import os
import csv
import copy
import argparse
import itertools
import time
import threading

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyttsx3

import tkinter as tk
from tkinter import ttk
from tkinter.ttk import *
from tkinter import StringVar, Radiobutton, PhotoImage
from PIL import Image, ImageTk, ImageDraw, ImageFilter
from itertools import count, cycle

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier

gender = 'Male'
# GIF Part
# ================== UI Part ======================
class ImageLabel(tk.Label):
    """
    A Label that displays images, and plays them if they are gifs
    :im: A PIL Image instance or a string filename
    """
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        frames = []

        try:
            for i in count(1):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        self.frames = cycle(frames)

        try:
            # self.delay = im.info['duration']
            self.delay = 10
        except:
            self.delay = 10

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()

    def unload(self):
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self.after(self.delay, self.next_frame)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=1050)
    parser.add_argument("--height", help="cap height", type=int, default=740)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=int,
        default=0.5,
    )

    args = parser.parse_args()

    return args

def main():
    global recognized_character, recognized_characters, recognized_characters_display,gender
    recognized_character = ''
    recognized_characters = ''
    recognized_characters_display = ''
    username = ''
    user_info = {'name':'Test','gender':'Male'}
    
    count = 5
    
    # Initialize Tkinter root window
    root = tk.Tk()
    root.title("Hand Gesture Recognition")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    window_width = 900
    window_height = 600
    
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.config(bg='')
    
    # Frames
    parentFrame = tk.Frame(root,bg='#0A364D')
    loading = tk.Frame(root, bg='#0A364D',width=200, height=100)
    # Info Form
    formFrame = tk.Frame(parentFrame, bg="#0A364D")
     # Main Frame of Project
    mainFrame = tk.Frame(parentFrame,bg='#0A364D')
   
    # Titlebar
    # root.overrideredirect(True)
    title_bar = tk.Frame(parentFrame,bg='#0A364D',relief='raised',bd=0,highlightthickness=0)
    titleName = tk.Label(title_bar,text='   American Sign Language Recognization System for Mutes',pady=3,padx=(window_width//2),justify='center',bg='#0A364D',fg='white',font=("Arial", 12))
    close_btn = tk.Button(title_bar,text=' X ',command=root.destroy,bg='#0A364D',pady=3,padx=5,activebackground='red',bd=0,font='bold',fg='white',highlightthickness=0)
    
    # title_bar.pack(fill="x")
    # close_btn.pack(side='right')
    
    # Window fucntions
    def move_window(event):
        root.geometry('+{0}+{1}'.format(event.x_root,event.y_root))
    
    def change_on_hovering(event):
        # global close_btn
        close_btn['bg'] = 'red'
        
    def return_to_normal(evetn):
        # global close_btn
        close_btn['bg'] = '#0A364D'
    
    title_bar.bind('<B1-Motion>',move_window)
    close_btn.bind('<Enter>',change_on_hovering)
    close_btn.bind('<Leave>',return_to_normal)
    
    
   
    loading.pack(fill='both', expand=1, pady=00, padx=00)
    
    # Create a label for the loading message
    loading_label = ttk.Label(loading,background='#0A364D', text="Loading...", font=("Arial", 20),borderwidth=1,foreground='white')
    loading_label.pack(pady=50)
    lbl = ImageLabel(loading,borderwidth=0)
    lbl.pack()
    lbl.load('./assets/loading1.gif')
    
    def countdown(count):
        if count >0:
            root.after(1000,countdown, count-1)
            print(count)
        if count == 1:
            parentFrame.pack(fill='both', expand=1)
            formFrame.pack(fill='both', expand=1)
            loading.pack_forget()
    
    countdown(count)
    
    
    formTitle = tk.Label(formFrame,background='#0A364D', foreground='white', text="Welcome to American Sign Language Recognization System", font=('Arial',23))
    b_image = Image.open("./assets/banner.jpg")
    b_image = b_image.resize((900,200))
    banner = ImageTk.PhotoImage(b_image)
    print(banner)
    bannerLabel = tk.Label(formFrame, image=banner)
    bannerLabel.pack(pady=20)
    
    formTitle.pack()
    
    # Input Fields
    # Styling
    label_font = ("Helvetica", 18)
    entry_font = ("Helvetica", 11)
    button_font = ("Helvetica", 12, "bold")
    button_bg = "#3c8ce7"
    button_fg = "#ffffff"
    # Title Label
    

    # Set up the name input
    tk.Label(formFrame, text="Enter Your Name:", font=("Helvetica", 14, "bold"), bg="#0A364D", fg="white").pack(pady=(10, 0))
    name_var = StringVar()
    name_entry = tk.Entry(formFrame,borderwidth=0, textvariable=name_var, font=("Helvetica", 18, "bold"), width=25, justify='center')
    name_entry.pack(pady=20)

    # Set up gender radio buttons
    tk.Label(formFrame, text="Select Gender:", font=("Helvetica", 14, "bold"), bg="#0A364D", fg="white").pack(pady=(20, 0))
    gender_var = StringVar(value="Male")
    gender_frame = tk.Frame(formFrame, bg="#0A364D")
    gender_frame.pack(pady=5)
    Radiobutton(gender_frame, text="Male", variable=gender_var, value="Male", font=entry_font, bg="#0A364D",fg='gray').pack(side="left", padx=5)
    Radiobutton(gender_frame, text="Female", variable=gender_var, value="Female", font=entry_font, bg="#0A364D",fg='gray').pack(side="left", padx=5)
    
    def on_submit():
        user_info["name"] = name_var.get()
        user_info["gender"] = gender_var.get()
        global username, gender
        username = user_info['name']
        gender = user_info['gender']
        label1.config(text=username+' is saying...')
        print(username)
        # mainFrame.pack(fill='both', expand=1)
        titleName.pack(side='left')
        
        mainFrame.pack()
        formFrame.pack_forget()
        
    # Submit Button
    btn_border = tk.Frame(formFrame,highlightbackground='white',highlightthickness=2,bd=0)
    submit_button = tk.Button(btn_border, text="Start Conversation", command=on_submit, font=button_font,borderwidth=0, bg='#0A364D', fg='white', width=15)
    submit_button.pack()
    btn_border.pack(pady=20)
    
    
    
    
   
    # Camera setup
    cap = cv.VideoCapture(0)  # Replace with your camera device or video input

    # Tkinter Canvas to display OpenCV frame
    canvas = tk.Label(mainFrame, width=500, height=350)
    canvas.pack(expand=1,pady=0)
    # Showing Recognized Text
    label1 = tk.Label(mainFrame, text=username, font=("Arial",14),bg='#0A364D',fg='white')
    label1.pack(pady=10)
    label = tk.Label(mainFrame, text='', font=("Arial",18),bg='#0A364D',fg='white')
    label.pack(pady=5)

    # Start and Clear buttons
    def capture_character():  
        global recognized_character,recognized_characters              
        recognized_characters += recognized_character
        recognized_character = ''
        label.config(text=recognized_characters)
        print(recognized_characters)
        
    def add_space():  
        global recognized_character,recognized_characters              
        recognized_characters += ' '
        recognized_character = ''
        label.config(text=recognized_characters)
        print(recognized_characters)
        

    def clear_text():
        global recognized_character,recognized_characters, recognized_characters_display
        recognized_characters = ""
        recognized_character = ""
        recognized_characters_display = ""
        label.config(text=recognized_characters)
        print("Cleared recognized text.")
    
    def delete_text():
        global recognized_character,recognized_characters, recognized_characters_display
        recognized_characters = recognized_characters[:-1]
        recognized_character = ""
        recognized_characters_display = ""
        label.config(text=recognized_characters)
        print("Cleared recognized text.")
        
    def text_to_speech():
        """Convert text to speech and play it."""
    # print(speakerGender)
        text = recognized_characters
        if text:
            try:
                    engine = pyttsx3.init()
                    # Set voice to female
                    voices = engine.getProperty('voices')
                    if gender == 'Male':
                        engine.setProperty('voice', voices[0].id)                
                    else:
                        engine.setProperty('voice', voices[1].id)                
                    engine.setProperty('rate', 125)                
                    engine.say(text)
                    engine.runAndWait()
            except Exception as e:
                        print(f"An error occurred: {e}")


    def on_key_press(event):
        if event.char == 'g' or event.char == 'G':
            capture_character()
        elif event.char == 'd' or event.char == 'D':
            delete_text()
        elif event.char == 'c' or event.char == 'C':
            clear_text()
        elif event.char == 's' or event.char == 'S':
            text_to_speech()
        elif event.char == ' ':
            add_space()
        print(event.char)
    
    root.bind("<Key>", on_key_press)

    capture_button = tk.Button(mainFrame, text="Capture (G/g)", font=("Helvetica", 14, "bold"), bg="#3c8ce7", fg="white", command=capture_character)
    capture_button.pack(side="left", padx=10)
    
    space_button = tk.Button(mainFrame, text="Space (space_bar)", font=("Helvetica", 14, "bold"), bg="red", fg="white", command=add_space)
    space_button.pack(side="left", padx=10)
    
    delete_button = tk.Button(mainFrame, text="Delete (D/d)", font=("Helvetica", 14, "bold"), bg="red", fg="white", command=delete_text)
    delete_button.pack(side="left", padx=10)
    
    clear_button = tk.Button(mainFrame, text="Clear (C/c)", font=("Helvetica", 14, "bold"), bg="red", fg="white", command=clear_text)
    clear_button.pack(side="left", padx=10)
    
    speak_button = tk.Button(mainFrame, text="Speak (S/s)", font=("Helvetica", 14, "bold"), bg="red", fg="white", command=text_to_speech)
    speak_button.pack(side="left", padx=10)


    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True
    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #  ########################################################################
    mode = 0

    # Function to process frames
    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Flip and process frame for display (adjust as needed for hand recognition)
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGRA2RGB)
            debug_image = copy.deepcopy(frame)
             # Process Key (ESC: end) #################################################
            key = cv.waitKey(1)
            # print(select_mode(key, mode))
            # number, mode = select_mode(key, mode)
            number, mode = 1,0
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    global recognized_character
                    recognized_character = keypoint_classifier_labels[hand_sign_id]
                    # recognized_characters += recognized_character
                    # print(recognized_character)
                    # Finger gesture classification
                    finger_gesture_id = 0

                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                    )
            
            img = Image.fromarray(debug_image)
            imgtk = ImageTk.PhotoImage(image=img)
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)
            
            # Placeholder for recognition process - Add recognition updates here
            recognized_characters_display = "Recognized Text Here"  # Replace with actual recognized characters

        # Repeat after a delay
        root.after(10, update_frame)

    # Start the video feed update
    update_frame()

    # Run the Tkinter event loop
    root.mainloop()
   
    # Release resources after Tkinter window is closed
    cap.release()
    cv.destroyAllWindows()

def select_mode(key, mode):
    number = -1
    if 65 <= key <= 90:  # A ~ B
        number = key - 65
    if key == 110:  # n (Inference Mode)
        mode = 0
    if key == 107:  # k (Capturing Landmark From Camera Mode)
        mode = 1
    if key == 100:  # d (Capturing Landmarks From Provided Dataset Mode)
        mode = 2
    return number, mode





def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if (mode == 1 or mode == 2) and (0 <= number <= 35):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[3]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[3]),
            tuple(landmark_point[4]),
            (255, 255, 255),
            2,
        )

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[6]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[6]),
            tuple(landmark_point[7]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[7]),
            tuple(landmark_point[8]),
            (255, 255, 255),
            2,
        )

        # Middle finger
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[10]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[10]),
            tuple(landmark_point[11]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[11]),
            tuple(landmark_point[12]),
            (255, 255, 255),
            2,
        )

        # Ring finger
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[14]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[14]),
            tuple(landmark_point[15]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[15]),
            tuple(landmark_point[16]),
            (255, 255, 255),
            2,
        )

        # Little finger
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[18]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[18]),
            tuple(landmark_point[19]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[19]),
            tuple(landmark_point[20]),
            (255, 255, 255),
            2,
        )

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[0]),
            tuple(landmark_point[1]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[1]),
            tuple(landmark_point[2]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[2]),
            tuple(landmark_point[5]),
            (255, 255, 255),
            2,
        )
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(
            image,
            tuple(landmark_point[5]),
            tuple(landmark_point[9]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[9]),
            tuple(landmark_point[13]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[13]),
            tuple(landmark_point[17]),
            (255, 255, 255),
            2,
        )
        cv.line(
            image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6
        )
        cv.line(
            image,
            tuple(landmark_point[17]),
            tuple(landmark_point[0]),
            (255, 255, 255),
            2,
        )

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  #
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  #
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(
        image,
        info_text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        # "FPS:" + str(fps),
        "Press Q/Esc to Exit...",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv.LINE_AA,
    )
    cv.putText(
        image,
        # "FPS:" + str(fps),
        "Press Q/Esc to Exit...",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    mode_string = [
        "Logging Key Point",
        "Capturing Landmarks From Provided Dataset Mode",
    ]
    if 1 <= mode <= 2:
        cv.putText(
            image,
            "MODE:" + mode_string[mode - 1],
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
        if 0 <= number <= 9:
            cv.putText(
                image,
                "NUM:" + str(number),
                (10, 110),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )
    return image



# Execute main
if __name__ == "__main__":
    main()