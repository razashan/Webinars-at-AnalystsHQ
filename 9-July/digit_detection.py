import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('digit_recognizer_model.h5')


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert the image to grayscale
    image = image.convert('L')
    # Normalize the pixel values to be in the range of 0 to 1
    image = np.array(image) / 255.0
    # Reshape the image to match the model's input shape
    image = image.reshape(1, 28, 28, 1)
    return image


# Function to handle button click event
def predict_image():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()

    if file_path:
        # Load the selected image
        image = Image.open(file_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make a prediction using the model
        predictions = model.predict(preprocessed_image)

        # Get the predicted label
        predicted_label = np.argmax(predictions[0])

        # Display the predicted label
        label.config(text=f"Predicted Label: {predicted_label}")

        # Display the image in the GUI
        image = image.resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=photo)
        canvas.image = photo


# Create the GUI window
window = tk.Tk()
window.title("Digit Recognition using CNN")
window.geometry("400x350")

# Create a title label
title_label = tk.Label(window, text="Digit Recognition using CNN", font=("Arial", 16, "bold"))
title_label.pack(pady=10)

# Create a canvas to display the image
canvas = tk.Canvas(window, width=200, height=200)
canvas.pack()

# Create a button to open the image file
button = tk.Button(window, text="Open Image", command=predict_image, font=("Arial", 12))
button.pack(pady=10)

# Create a label to display the predicted label
label = tk.Label(window, text="Predicted Label: ", font=("Arial", 14))
label.pack(pady=10)

# Start the GUI event loop
window.mainloop()
