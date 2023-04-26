import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
import pickle
import time
from streamlit_drawable_canvas import st_canvas

# Charger le modèle entraîné
with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

# Fonction qui transforme l'image du dessin en un tableau numpy
def transform_image(img):
    # Redimensionner l'image à une taille fixe
    img = img.resize((28, 28))
    # Convertir l'image en niveaux de gris
    img = img.convert('L')
    # Convertir l'image en un tableau numpy
    img = np.array(img)
    # Inverser les couleurs (0 pour le trait, 255 pour le fond)
    img = 255 - img
    # Normaliser les valeurs de pixel entre 0 et 1
    img = img / 255.0
    # Aplatir l'image en un vecteur 1D
    img = img.reshape(1, 28*28)
    return img

# Fonction qui prédit le chiffre dessiné et renvoie le résultat
def predict_number(img):
    # Transformer l'image du dessin en un tableau numpy
    img = transform_image(img)
    # Prédire le chiffre dessiné en utilisant le modèle
    prediction = model.predict(img)
    return prediction[0]

# Fonction qui affiche l'interface graphique
def game():
    # Initialiser les variables
    canvas_size = 300
    canvas_color = '#ffffff'
    drawing_color = '#000000'
    n_tries = 10
    n_correct = 0
    n_incorrect = 0
    game_over = False
    current_number = None

    # Afficher le titre et les instructions
    st.title("Jeu de dessin")
    st.write("Dessinez un chiffre entre 0 et 9 et cliquez sur le bouton pour prédire le chiffre dessiné.")

    # Créer le canvas de dessin
    # fig, ax = plt.subplots()
    # ax.set_xlim([0, canvas_size])
    # ax.set_ylim([canvas_size, 0])
    # canvas = np.zeros((canvas_size, canvas_size))

    # Créer le canvas de dessin
    canvas = st_canvas(
        fill_color=canvas_color,
        stroke_width=20,
        stroke_color=drawing_color,
        background_color=canvas_color,
        width=canvas_size,
        height=canvas_size,
        drawing_mode="freedraw",
        key="canvas"
    )


    # Afficher le canvas de dessin
    # ax.imshow(canvas, cmap='gray', origin='upper', extent=[0, canvas_size, 0, canvas_size])
    # st.pyplot(fig)
    # Créer un bouton pour la prochaine tentative
    next_attempt_button = st.button("Tentative suivante", key="next_attempt")
    predict_button_key = 0
    # Boucle principale du jeu
    while not game_over:

        # Vérifier si le chiffre actuel a été prédit
        if current_number is not None:
            st.write("Vous avez dessiné le chiffre :", current_number)
            # Mettre à jour les statistiques
            if current_number == n_tries:
                n_correct += 1
            else:
                n_incorrect += 1
            n_tries -= 1
            current_number = None
            # Vérifier si le jeu est terminé
            if n_tries == 0:
                game_over = True
                # Afficher les statistiques
                st.write("Le jeu est terminé !")
                st.write("Vous avez eu", n_correct, "bonnes réponses et", n_incorrect, "mauvaises réponses.")
                break
            else:
                # Afficher le bouton pour passer à la prochaine tentative
                if next_attempt_button:
                # Réinitialiser le canvas de dessin
                    canvas = np.zeros((canvas_size,canvas_size))

        # Vérifier si le bouton de prédiction a été cliqué
        if not game_over:
            if st.button("Prédire", key=f"predict_button_{predict_button_key}"):
                # Convertir le canvas en une image
                img = Image.fromarray((canvas * 255).astype(np.uint8), mode='L')
                # Prédire le chiffre dessiné
                current_number = predict_number(img)
            predict_button_key += 1
        # Attendre 0.1 seconde pour éviter d'utiliser trop de ressources
        time.sleep(0.1)

    # Afficher un bouton de réinitialisation pour recommencer le jeu
    if st.button("Recommencer", key="restart"):
        game()

# Appeler la fonction pour lancer le jeu
game()
