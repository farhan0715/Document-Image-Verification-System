import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import face_recognition
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8s-seg.pt")  # Adjust with your model path

class DocumentVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Photo Verification System")
        self.root.geometry("1200x800")
        self.bg_color = '#e0f7fa'  # Set background color of the page

        # Set the background color of the main window
        self.root.config(bg=self.bg_color)

        # Frame for heading
        self.frame_heading = tk.Frame(root, bg=self.bg_color)
        self.frame_heading.pack(side=tk.TOP, pady=(20, 10))

        self.heading_label = tk.Label(self.frame_heading, text="Welcome to my Document Photo Verification App", 
                                      font=('Arial', 20, 'bold'), bg=self.bg_color)
        self.heading_label.pack()

        # Frame for buttons and result text
        self.frame_top = tk.Frame(root, bg=self.bg_color)
        self.frame_top.pack(side=tk.TOP, pady=10)

        self.load_passport_button = tk.Button(self.frame_top, text="Select Document Image", command=self.load_passport_image,
                                             bg='#4CAF50', fg='white')
        self.load_passport_button.grid(row=0, column=0, padx=10)

        self.load_person_button = tk.Button(self.frame_top, text="Select Person Image", command=self.load_person_image,
                                           bg='#2196F3', fg='white')
        self.load_person_button.grid(row=0, column=1, padx=10)

        self.reset_button = tk.Button(self.frame_top, text="Reset", command=self.reset_all, 
                                      bg='#FF5722', fg='white')
        self.reset_button.grid(row=1, column=1, padx=10, pady=10)

        self.verify_button = tk.Button(self.frame_top, text="Verify", command=self.verify_person,
                                       bg='#FFC107', fg='black')
        self.verify_button.grid(row=0, column=2, padx=10)

        self.result_label = tk.Label(self.frame_top, text="Result: ", font=('Arial', 12, 'bold'), bg=self.bg_color)
        self.result_label.grid(row=0, column=3, padx=10)

        # Frame for images
        self.frame_images = tk.Frame(root, bg=self.bg_color)
        self.frame_images.pack(side=tk.TOP, pady=10)

        self.cropped_label = tk.Label(self.frame_images, bg=self.bg_color)
        self.cropped_label.grid(row=0, column=0, padx=10)

        self.passport_label = tk.Label(self.frame_images, bg=self.bg_color)
        self.passport_label.grid(row=0, column=1, padx=10)

        self.person_label = tk.Label(self.frame_images, bg=self.bg_color)
        self.person_label.grid(row=0, column=2, padx=10)

        # Labels to display the names of the images
        self.cropped_name_label = tk.Label(self.frame_images, text="", font=('Arial', 10), bg=self.bg_color)
        self.cropped_name_label.grid(row=1, column=0, padx=10)

        self.passport_name_label = tk.Label(self.frame_images, text="", font=('Arial', 10), bg=self.bg_color)
        self.passport_name_label.grid(row=1, column=1, padx=10)

        self.person_name_label = tk.Label(self.frame_images, text="", font=('Arial', 10), bg=self.bg_color)
        self.person_name_label.grid(row=1, column=2, padx=10)

        # Initialize image variables
        self.passport_image = None
        self.person_image = None
        self.cropped_person = None
        self.passport_image_name = "No Document image loaded"
        self.person_image_name = "No person image loaded"
        self.cropped_image_name = "No cropped image"

    def load_passport_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.passport_image = cv2.imread(file_path)
            self.passport_image_rgb = cv2.cvtColor(self.passport_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.passport_image_rgb, self.passport_label)
            self.passport_image_name = file_path.split('/')[-1]  # Get the file name
            self.passport_name_label.config(text=self.passport_image_name)

    def load_person_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.person_image = cv2.imread(file_path)
            self.person_image_rgb = cv2.cvtColor(self.person_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.person_image_rgb, self.person_label)
            self.person_image_name = file_path.split('/')[-1]  # Get the file name
            self.person_name_label.config(text=self.person_image_name)

    def display_image(self, img, label):
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((400, 400))  # Resize the image to fit the label
        img_tk = ImageTk.PhotoImage(image=img_pil)
        label.img_tk = img_tk  # keep reference
        label.config(image=img_tk)

    def verify_person(self):
        if self.passport_image is None or self.person_image is None:
            self.result_label.config(text="Result: Please load both images.", fg='red')
            return

        results = model.predict(source=self.passport_image, conf=0.45)
        person_class_index = [k for k, v in model.names.items() if v == 'person'][0]

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.cls.item() == person_class_index:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    self.cropped_person = self.passport_image[y1:y2, x1:x2]

                    # Display the cropped image for debugging
                    cropped_person_rgb = cv2.cvtColor(self.cropped_person, cv2.COLOR_BGR2RGB)
                    self.display_image(cropped_person_rgb, self.cropped_label)
                    self.cropped_image_name = "Cropped image from Document"
                    self.cropped_name_label.config(text=self.cropped_image_name)

                    cropped_person_encoding = face_recognition.face_encodings(cropped_person_rgb)
                    input_person_encoding = face_recognition.face_encodings(self.person_image_rgb)

                    if len(cropped_person_encoding) > 0 and len(input_person_encoding) > 0:
                        match = face_recognition.compare_faces([cropped_person_encoding[0]], input_person_encoding[0])

                        if match[0]:
                            self.result_label.config(text="Result: Verified Person", fg='green')
                        else:
                            self.result_label.config(text="Result: Unverified Person", fg='red')
                    else:
                        self.result_label.config(text="Result: Could not find faces in one or both images.", fg='red')
                    return

        self.result_label.config(text="Result: No person detected in the Document image.", fg='red')

    def reset_all(self):
        self.passport_image = None
        self.person_image = None
        self.cropped_person = None
        self.passport_label.config(image='')
        self.person_label.config(image='')
        self.cropped_label.config(image='')
        self.result_label.config(text="Result: ", fg='black')
        self.passport_image_name = "No Document image loaded"
        self.person_image_name = "No person image loaded"
        self.cropped_image_name = "No cropped image"
        self.passport_name_label.config(text=self.passport_image_name)
        self.person_name_label.config(text=self.person_image_name)
        self.cropped_name_label.config(text=self.cropped_image_name)

# Create the main window
root = tk.Tk()
app = DocumentVerificationApp(root)
root.mainloop()
