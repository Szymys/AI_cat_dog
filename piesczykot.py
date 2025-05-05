
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# TKINTER DO APLIKACJI OKIENKOWEJ
import tkinter as tk
from tkinter import filedialog, messagebox


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
    )

train_generator = train_datagen.flow_from_directory(
    './data/train', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary', 
    subset='training'
    )

validation_generator = train_datagen.flow_from_directory(
    './data/train', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='binary', 
    subset='validation'
    )



model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

     
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


model.fit(
    train_generator, 
    epochs=15, 
    validation_data=validation_generator
    )


def test_model(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0  
    
    prediction = model.predict(img_array)  
    
    if prediction < 0.5:
        result = 'To jest kot'
    else:
        result = 'To jest pies'
    
    
    messagebox.showinfo("Wynik", result)


def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if filepath:
        test_model(filepath)


root = tk.Tk()
root.title("Aplikacja: Pies czy Kot")
root.geometry("400x300")
root.config(bg="#f0f0f0")  


title_label = tk.Label(root, text="Wybierz zdjęcie do analizy", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)


btn_open = tk.Button(root, text="Wybierz zdjęcie", command=open_file, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=20, pady=10)
btn_open.pack(pady=40)


btn_exit = tk.Button(root, text="Zamknij", command=root.quit, font=("Helvetica", 12), bg="#f44336", fg="white", relief="flat", padx=20, pady=10)
btn_exit.pack(pady=10)


root.mainloop()
