import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pickle
import uuid





######################################################################################################
# Charger le modèle entraîné
with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

# Configurer l'application Streamlit
st.set_page_config(page_title="Number Recognition Game", page_icon=":pencil2:", layout="wide")

st.title("Number Recognition")

canvas_size = 300
predictions = []
n_prediction = st.session_state.get('n_prediction', 0)
score = st.session_state.get('score', 0)
max_try = 10
game_over = False
try_left = st.session_state.get('try_left', 10)


canvas = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=canvas_size,
    width=canvas_size,
    drawing_mode="freedraw",
    key="canvas"
)

true_number = st.selectbox("Veuillez saisir le chiffre que vous allez dessiner", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
predict_button = st.button('Predict', key=f"predict")

def pred_model():
    predictions = []
    img_resized = Image.fromarray(canvas.image_data.astype('uint8')).resize((28, 28))

    # Convert the image to grayscale
    img_gray = img_resized.convert('L')

    # Convertir l'image en array numpy
    img_array = np.array(img_gray)

    # Traiter l'image comme nécessaire (ex: la normaliser)
    processed_img_array = img_array / 255.0

    st.image(processed_img_array)
    # Stocker l'image dans une variable
    image = np.expand_dims(processed_img_array, axis=0)

    # Prédire le chiffre en utilisant le modèle
    prediction = model.predict(image)[0]

    # Ajouter la prédiction à la liste de prédictions
    predictions.append(np.argmax(prediction))

    return predictions

def test():
    global n_prediction, score, try_left
    predictions = pred_model()
    # Afficher le résultat de la prédiction

    st.write(f'La prédiction du chiffre est {np.argmax(predictions)}')

    # Incrémenter le compteur de prédictions
    n_prediction += 1
    try_left -= 1

    # Vérifier si la prédiction est correcte
    if np.argmax(predictions) == true_number:
        score += 1
        st.write(f"Bonne réponse !")
    else:
        st.write("Mauvaise réponse.")

    # Stocker les nouvelles valeurs dans st.session_state
    st.session_state['n_prediction'] = n_prediction
    st.session_state['score'] = score
    st.session_state['try_left'] = try_left


################################################################################

def play():
    global n_prediction, score, try_left, game_over
    # Prédire le chiffre dessiné par l'utilisateur

    if predict_button:
        test()
        st.write(f"il vous reste {try_left} essaies")
        st.write(f"n_prediction est à {n_prediction}")
        st.write(f"Votre score est de {score}")

    else :
        st.write("appuyer sur le bouton predict !")

    if n_prediction == max_try:
        # Calculer le score final
        score_ratio = score / max_try

        # Afficher les statistiques
        st.write("Le jeu est terminé !")
        st.write(f"Vous avez fait {max_try} tentatives, et votre score est de {score}/{max_try}.")
        st.write(f"Votre ratio de bonnes réponses est de {score_ratio:.2f}.")



play()
