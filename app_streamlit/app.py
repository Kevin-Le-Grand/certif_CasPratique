import streamlit as st
import cv2
import numpy as np
import json
import joblib
import tensorflow as tf
from PIL import Image
from functions import *

# Configuration de la mise en page
st.set_page_config(layout='wide')


col1,col2 = st.columns(2)
with col1:
    st.title("Application d'Authentification")
with col2:
    image = st.file_uploader("Sélectionnez une image", type=['jpg', 'png', 'jpeg', 'bmp'])



# Charger les modèles et encodeurs
encodeur_OD = joblib.load('../code/joblib/encodeur_OD.joblib')
encodeur_OG = joblib.load('../code/joblib/encodeur_OG.joblib')
encodeur_ODG = joblib.load('../code/joblib/encodeur_ODG.joblib')

model_OG = tf.keras.models.load_model('../code/OG_classif')
model_OG.trainable = False
model_OD = tf.keras.models.load_model('../code/OD_classif')
model_OD.trainable = False
model_ODG = tf.keras.models.load_model('../code/ODG_classif')
model_ODG.trainable = False


if image is not None:
    image = Image.open(image)
    image = np.array(image)

    # Prétraiter l'image
    image_prep = preprocess_img(image)

    # Créer deux colonnes
    col1, col2, col3 = st.columns(3)

    # Afficher l'image dans la première colonne
    with col1:
        st.image(image, caption='Image sélectionnée', use_column_width=True)

    # Prédiction et affichage des résultats dans la deuxième colonne
    with col2:
        detect_eye, fiabilite = detect_ODG(image_prep)

        if detect_eye == 0:
            oeil = "droit"
            probs = model_OD.predict(np.array([image_prep]))
            prediction_user = np.argmax(probs)
            decode_prediction_user = encodeur_OD.inverse_transform([prediction_user])
        else:
            oeil = "gauche"
            probs = model_OG.predict(np.array([image_prep]))
            prediction_user = np.argmax(probs)
            decode_prediction_user = encodeur_OG.inverse_transform([prediction_user])

        nom, annee_embauche, genre, poste = recherche_ID(decode_prediction_user[0])

        table_data = [
            ("ID", decode_prediction_user[0]),
            ("Nom", nom),
            ("Année d'embauche", annee_embauche),
            ("Genre", genre),
            ("Poste", poste)
        ]

        # Définir le style CSS pour la colonne de gauche (libellés)
        left_column_style = "font-size:16px; text-align:left; padding:5px; background-color:black;"

        # Définir le style CSS pour la colonne de droite (valeurs)
        right_column_style = "font-size:20px; text-align:left; padding:5px;"

        # Créer une chaîne de caractères HTML pour le tableau personnalisé
        # Créer une chaîne de caractères HTML pour le tableau personnalisé centré
        table_html = "<table style='margin: 0 auto;'>"
        for label, value in table_data:
            table_html += f"<tr><td style='{left_column_style}'>{label}</td><td style='{right_column_style}'>{value}</td></tr>"
        table_html += "</table>"


        # Afficher le texte et le tableau dans la deuxième colonne
        st.markdown(f"<p style='text-align:center; background-color:green; padding:10px; font-size:20px; color:white;'>Oeil {oeil} détecté avec une fiabilité de {fiabilite}%</p>", unsafe_allow_html=True)
        st.markdown(table_html, unsafe_allow_html=True)

    with col3:
        fig = draw_bar_chart(probs, encodeur_OD, prediction_user)
        st.pyplot(fig)
