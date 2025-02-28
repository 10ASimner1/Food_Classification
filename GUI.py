import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import os
from Helper_Function import load_and_prep_image
import tensorflow_datasets as tfds

# Load trained model
model_path = "Food_Vision_efficientnetb1_feature_extract_model"
try:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Model not found at: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load class names from Food101 dataset
datasets_list = tfds.list_builders()
target_dataset = "food101"
if target_dataset in datasets_list:
    _, ds_info = tfds.load(
        name="food101",
        split=["train[:1%]"],  # Only need info, not actual data
        shuffle_files=False,
        as_supervised=True,
        with_info=True,
    )
    class_names = ds_info.features["label"].names
else:
    print(f"Dataset '{target_dataset}' not found.")
    class_names = None

def predict_and_display(image_path):
    """Predicts and displays the image with prediction."""
    if model is None or class_names is None:
        print("Model or class names not loaded. Cannot predict.")
        return

    img = load_and_prep_image(image_path, scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_class = class_names[pred_prob.argmax()]

    # Display image in GUI
    pil_img = Image.open(image_path)
    pil_img = pil_img.resize((300, 300), Image.LANCZOS)  # Resize for display
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img  # Keep a reference!

    # Display prediction
    prediction_label.config(text=f"Prediction: {pred_class}\nProbability: {pred_prob.max():.2f}")

def browse_image():
    """Opens file dialog to select an image."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        predict_and_display(file_path)

# GUI setup
root = tk.Tk()
root.title("Food Image Classifier")

# Center the window
window_width = 400 
window_height = 550 # increased height to prevent cutoff

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_coordinate = int((screen_width/2) - (window_width/2))
y_coordinate = int((screen_height/2) - (window_height/2))

root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

style = ttk.Style()
style.theme_use("clam") 

# Create a frame for better layout management
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# Browse button
browse_button = ttk.Button(main_frame, text="Browse Image", command=browse_image, padding=10)
browse_button.pack(pady=20)

# Image display label
image_label = ttk.Label(main_frame)
image_label.pack(pady=10)

# Prediction label with better styling
prediction_label = ttk.Label(main_frame, text="", font=("Arial", 12))
prediction_label.pack(pady=20, fill=tk.X) # fill X to expand label horizontally.

# Center the widgets in the main frame
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)
main_frame.rowconfigure(1, weight=1)
main_frame.rowconfigure(2, weight=1)

root.mainloop()
