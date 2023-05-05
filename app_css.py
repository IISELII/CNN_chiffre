import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Configurer l'application Streamlit
st.set_page_config(page_title="NB_RECO", page_icon=":pencil2:", layout="wide")


# charger le css
# with open('style.css') as css:
#         st.markdown(f'<style>{css.read}</style>', unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as c:
        st.markdown(f'<style>{c.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Charger le modèle entraîné
model = keras.models.load_model('modeloo.h5')


######################################################################################################

# Initialiser la variable de session pour le choix de la page
# if 'page_choice' not in st.session_state:
#     st.session_state.page_choice = 'ACCEUIL'



# Créer la barre de menu avec st.sidebar
# menu = ['ACCEUIL', 'GAME 1', 'GAME 2']
# choice = st.sidebar.selectbox('CHOISISSEZ UNE PAGE', menu, index=menu.index(st.session_state.page_choice))

# the side bar that contains radio buttons for selection of game
# with st.sidebar:
#     game = st.radio('SELECT A GAME',
#     ('ACCEUIL', 'GAME 1', 'GAME 2'),
#     index=('ACCEUIL', 'GAME 1', 'GAME 2').index(st.session_state.page_choice))


# Stocker la valeur de la page sélectionnée dans la variable de session
# st.session_state.page_choice = choice if choice != 'ACCEUIL' else game


sample = pd.read_csv('data/sample.csv')
X_s = sample.drop(["label"], axis = 1)
X_s = X_s.values.reshape(-1, 28, 28, 1)


with st.container():

    with st.container():
        selected = option_menu(
            menu_title=None,
            options=["ACCEUIL", "GAME 1", "GAME 2"],
            icons=['house', 'cloud-upload', "graph-up-arrow"],
            menu_icon="cast",
            orientation="horizontal",
            styles={
                "nav-link": {
                    "text-align": "left",
                    "--hover-color": "#ffc107",
                },

                "nav-link-selected": {"background-color": "#ffc107"},


            }
        )

        if selected == "ACCEUIL":
            st.subheader('ACCEUIL')
            st.title('Bienvenue !')
            st.header('Tester notre application et tester les prédictions de notre modèle !')
            successive_outputs = [layer.output for layer in model.layers[0:]]
            visualization_model = keras.models.Model(inputs = model.input, outputs = successive_outputs)
            test = ((X_s).reshape((-1,28,28,1)))/255.0
            successive_feature_maps = visualization_model.predict(test)
            layer_names = [layer.name for layer in model.layers]
            for layer_name, feature_map in zip(layer_names, successive_feature_maps):
                if len(feature_map.shape) == 4:
                        n_features = feature_map.shape[-1]
                        size = feature_map.shape[ 1]
                        display_grid = np.zeros((size, size * n_features))
                        for i in range(n_features):
                                x  = feature_map[-1, :, :, i]
                                x -= x.mean()
                                x /= x.std ()
                                x *=  64
                                x += 128
                                x  = np.clip(x, 0, 255).astype('uint8')
                                display_grid[:, i * size : (i + 1) * size] = x
                                scale = 20. / n_features
                        fig = plt.figure( figsize=(scale * n_features, scale) )
                        plt.title ( layer_name )
                        plt.grid  ( False )
                        plt.imshow( display_grid, aspect='auto', cmap='viridis' )
                        st.pyplot(fig)



        if selected == "GAME 1":
            st.title('Number Recognition')

            # Function to preprocess the image
            def preprocess_image(image):
                # Convert the image to grayscale
                image = image.convert('L')
                # Resize the image to the required input shape of the model
                image = image.resize((28, 28))
                # Invert the pixel values
                image = np.invert(image)
                # Reshape the image to a 4D array with a batch size of 1
                image = np.reshape(image, (1, 28, 28, 1))
                # Normalize the pixel values
                image = image / 255.0
                return image

            # Create a file uploader widget
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)

                # Resize the image to a width of 300 pixels and proportional height
                width, height = image.size
                new_width = 600
                new_height = 600
                resized_image = image.resize((new_width, new_height))

                st.image(resized_image, caption='Uploaded Image', use_column_width=False)

                # Preprocess the image
                preprocessed_image = preprocess_image(image)

                # Use the model to predict the number in the image
                prediction = model.predict(preprocessed_image)
                predicted_number = np.argmax(prediction)

                # Display the predicted number
                st.header(f"Predicted number is: {predicted_number}")

            else:
                st.write("Please upload an image file")

        if selected == "GAME 2":

            if "init" not in st.session_state:
                st.session_state.init = True

            # Game 2
            st.title('Number Recognition')
            canvas_size = 300
            predictions = []
            n_prediction = st.session_state.get('n_prediction', 0)
            max_try = 5
            game_over = False
            try_left = st.session_state.get('try_left', 5)

            canvas = st_canvas(
                fill_color="black",
                stroke_width=10,
                stroke_color="white",
                background_color="black",
                height=300,
                width=300,
                drawing_mode="freedraw",
                key="canvas"
            )



            def pred_model():
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
                prediction = model.predict(image)

                st.write(prediction)
                # Ajouter la prédiction à la liste de prédictions
                predictions.append(np.argmax(prediction))

                st.session_state['predictions'] = predictions

                return predictions


            if canvas is not None:

                if 'key_b' not in st.session_state:
                    st.session_state['key_b'] = 0
                if 'key_g' not in st.session_state:
                    st.session_state['key_g'] = 0
                if 'score' not in st.session_state:
                    st.session_state['score'] = 0

                if n_prediction == max_try:
                    # Calculate the final score
                    score_ratio = st.session_state['score'] / max_try

                    # Display the statistics
                    st.header("The game is over!")
                    st.header(f"Your score ratio is {score_ratio:.2f}.")

                    restart = st.button("Restart", key='restart')

                    if restart:
                        "restart the page"
                        # Remove the stored values from st.session_state
                        st.session_state.pop('score', None)
                        st.session_state.pop('n_prediction', None)
                        st.session_state.pop('try_left', None)
                        st.session_state.pop('key_b', None)
                        st.session_state.pop('key_g', None)
                        st.experimental_rerun()

                predict_button = st.session_state.get('predict_button', False)

                if predict_button:

                    col3, col4 = st.columns(2)
                    with col3:
                        predictions = pred_model()

                        # Increment the prediction counter
                        n_prediction += 1
                        try_left -= 1

                    with col4:
                        st.write('')
                        st.write(f"Your number is: {predictions[0]}!")
                        st.write(f'You have {try_left} tries left')

                        # Store the new values in st.session_state
                        st.session_state['n_prediction'] = n_prediction
                        st.session_state['try_left'] = try_left

                        # Reset the 'predict_button' state
                        st.session_state['predict_button'] = False



                col1, col2 = st.columns(2)
                with col1:
                    st.write("Good prediction ?")
                    # Add buttons for correct and incorrect predictions
                    if st.button('Good', key=f"g{st.session_state['key_g']}"):
                        st.session_state['score'] += 1
                        st.session_state['key_g'] += 1


                with col2:
                    st.write("Bad prediction ? ")
                    if st.button('Bad', key=f"b{st.session_state['key_b']}"):
                        st.write('Bad prediction :(')
                        st.session_state['key_b'] += 1






# Check if the prediction is correct
                    # if st.button('Bad', key=f"b{key_b}"):
                    #     st.write('Too bad :(')

                    # if st.button('Good', key=f"g{key_g}"):
                    #     score += 1
                    #     st.write(f'{score}')
                    #     st.write('Great job!')
