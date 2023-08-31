import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from functions import * # Remplacez par l'importation de votre bibliothèque ML et de votre modèle


import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from PIL import Image
from keras.utils import to_categorical
import cv2

class ImagePredictionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Application d'Authentification")

        self.load_model()  # Chargez votre modèle ici



        self.select_button = tk.Button(root, text="Sélectionner une image", command=self.load_image)
        self.select_button.pack()
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        self.predict_button = tk.Button(root, text="Lancer la Prédiction", command=self.predict_image)
        self.predict_button.pack()
        
        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack()

    def load_model(self):
        # Chargez votre modèle ici en utilisant votre bibliothèque ML
        self.model = tf.keras.models.load_model("vgg16_side_OG2ID_classif")
        self.model.trainable = False

        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.image_prep=preprocess_img(self.image)  # Redimensionnez l'image pour l'affichage
            self.imagepath_label.config(text=f"Image : {file_path}")
            self.photo = ImageTk.PhotoImage(Image.open(file_path))
            self.imagedisp_label.config(image=self.photo)
    def predict_image(self):
        if hasattr(self, 'image'):
            # Prétraitez l'image si nécessaire avant de faire une prédiction
            # image_array = np.array(self.image)
            # image_array = tf.image.resize(image_array, (224, 224))
            
            # Faites la prédiction en utilisant votre modèle
            prediction = self.model.predict(np.array([self.image_prep]))  # Remplacez par la méthode de prédiction de votre modèle
            self.prediction_label.config(text=f"Prédiction du modèle : {prediction}")
        else:
            self.prediction_label.config(text="Aucune image sélectionnée")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictionApp(root)
    root.mainloop()
