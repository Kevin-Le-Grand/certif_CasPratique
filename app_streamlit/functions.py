import cv2
import numpy as np
import json
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

encodeur_OD = joblib.load('../code/joblib/encodeur_OD.joblib')
encodeur_OG = joblib.load('../code/joblib/encodeur_OG.joblib')
encodeur_ODG = joblib.load('../code/joblib/encodeur_ODG.joblib')

model_OG = tf.keras.models.load_model('../code/OG_classif')
model_OG.trainable = False
model_OD = tf.keras.models.load_model('../code/OD_classif')
model_OD.trainable = False
model_ODG = tf.keras.models.load_model('../code/ODG_classif')
model_ODG.trainable = False

## Prétraitement de l'image : redimensionnemennt et standardization 
def preprocess_img(img,new_dim=(240,320)):
    new_img=cv2.resize(img, (new_dim[1],new_dim[0]), interpolation = cv2.INTER_AREA)
    new_img = new_img/255
    return new_img

def recherche_ID(prediction_user):
    with open('../code/employees_info.json', 'r') as json_file:
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
    fiabilite = round(float(probs[0][prediction]),4)
    fiabilite *=100
    return decode_prediction, fiabilite

def draw_bar_chart(probs, encodeur, prediction_user):
    top_indices = np.argsort(probs[0])[-3:]
    id_1 = encodeur.inverse_transform([top_indices[0]])
    id_2 = encodeur.inverse_transform([top_indices[1]])
    id_3 = encodeur.inverse_transform([top_indices[2]])
    fiabilite_id3= round(float(probs[0][top_indices[-1]]),4)*100
    fiabilite_id2= round(float(probs[0][top_indices[-2]]),4)*100
    fiabilite_id1= round(float(probs[0][top_indices[-3]]),4)*100
    data = {
        f'ID {id_3[0]}': fiabilite_id3,
        f'ID {id_2[0]}': fiabilite_id2,
        f'ID {id_1[0]}': fiabilite_id1,
    }
    
    categories = list(data.keys())
    values = list(data.values())
    
    # Créer une figure Matplotlib
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    bars = ax.bar(categories, values)
    ax.set_ylabel('Valeurs')
    ax.set_xlabel('ID des employés')
    ax.set_title("Probabilités d'identification")

    # Masquer la ligne du haut du cadre
    ax.spines['top'].set_visible(False)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 2, round(yval, 2), ha='center', va='bottom')

    return fig