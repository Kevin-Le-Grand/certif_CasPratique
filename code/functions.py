import cv2
import numpy as np
import json
import joblib
import tensorflow as tf

encodeur_OD = joblib.load('./joblib/encodeur_OD.joblib')
encodeur_OG = joblib.load('./joblib/encodeur_OG.joblib')
encodeur_ODG = joblib.load('./joblib/encodeur_ODG.joblib')

model_OG = tf.keras.models.load_model('OG_classif')
model_OG.trainable = False
model_OD = tf.keras.models.load_model('OD_classif')
model_OD.trainable = False
model_ODG = tf.keras.models.load_model('ODG_classif')
model_ODG.trainable = False

## Prétraitement de l'image : redimensionnemennt et standardization 
def preprocess_img(img,new_dim=(240,320)):
    new_img=cv2.resize(img, (new_dim[1],new_dim[0]), interpolation = cv2.INTER_AREA)
    new_img = new_img/255
    return new_img

def recherche_ID(prediction_user):
    with open('employees_info.json', 'r') as json_file:
        data = json.load(json_file)
    info = data[str(prediction_user)]
    nom = info['nom']
    annee_embauche = info['annee_embauche']
    genre = info['genre']
    poste = info['poste']
    return nom,annee_embauche,genre,poste

def detect_ODG(image):
    probs=model_ODG.predict(np.array([image]))
    prediction = np.argmax(probs)             
    decode_prediction = encodeur_ODG.inverse_transform([prediction])
    return decode_prediction
