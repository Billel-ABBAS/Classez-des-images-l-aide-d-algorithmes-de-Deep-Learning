import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle Xception
model_xception = tf.keras.models.load_model('model/xception_best_model.keras')

# Affichage du titre avec un style amélioré (couleur, taille, alignement)
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Prédiction de la race de chiens avec Xception</h1>", unsafe_allow_html=True)

# Téléchargement d'image par l'utilisateur
uploaded_file = st.file_uploader("Choisir une image de chien...", type="jpg")

if uploaded_file is not None:
    # Afficher l'image téléchargée et la centrer automatiquement avec use_column_width=True
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée.', use_column_width=True, width=400)  # Cette option centre l'image

    
    # Bouton pour déclencher la prédiction
    if st.button('Prédire'):

        # Redimensionner l'image à la taille attendue par Xception (299, 299)
        image = image.resize((299, 299))
        
        # Prétraitement de l'image
        img = img_to_array(image)
        img = np.expand_dims(img, axis=0)  # Ajouter une dimension pour correspondre à l'entrée du modèle
        img = img / 255.0  # Normaliser l'image

        # Prédiction avec le modèle Xception
        prediction_xception = model_xception.predict(img)

        # Récupérer la classe prédite
        predicted_class_xception = np.argmax(prediction_xception)

        # Liste des noms de races mise à jour
        breed_names = ['Silky Terrier', 'Golden Retriever', 'German Shepherd']

        # Nom prédit par le modèle
        predicted_name = f"Nom prédit par Xception : {breed_names[predicted_class_xception]}"

        # Affichage formaté avec des tailles de police plus grandes
        st.markdown(f"<h2 style='text-align: center; color: blue;'>{predicted_name}</h2>", unsafe_allow_html=True)





