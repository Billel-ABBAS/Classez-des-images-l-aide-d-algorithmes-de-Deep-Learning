# Importations des bibliothèques TensorFlow et Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
                                     GlobalAveragePooling2D, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, Xception
from keras import layers
from keras import initializers
import keras
keras.utils.set_random_seed(812)
tf.config.experimental.enable_op_determinism()

# Importations pour la manipulation d'images et graphiques
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importations pour les métriques de modèles
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support



### Partie prétraitement

# Fonction fusionnée pour charger, redimensionner et afficher les images
def load_and_display_images(image_paths, titles, target_size=None):
    """
    Charge, redimensionne et affiche une série d'images dans une grille avec les titres correspondants.
    
    Parameters:
    - image_paths: Liste des chemins des images à afficher.
    - titles: Liste des titres correspondant aux images (à afficher sous chaque image).
    - target_size: Taille à laquelle redimensionner les images (largeur, hauteur).
    """
    # Créer une figure pour afficher les images
    plt.figure(figsize=(15, 5))

    # Boucle pour charger, redimensionner et afficher les images
    for i, image_path in enumerate(image_paths):
        # Charger et redimensionner l'image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)  # Convertir l'image en tableau de pixels
        
        # Afficher la confirmation dans la console
        print(f"L'image du {titles[i]} a été chargée avec la forme {img_array.shape}")

        # Afficher l'image
        plt.subplot(1, len(image_paths), i + 1)  # Positionner l'image sur une grille (1 ligne et N colonnes)
        plt.imshow(np.uint8(img_array))  # Afficher l'image redimensionnée
        plt.title(titles[i])  # Ajouter le titre correspondant à chaque image
        plt.axis('off')  

    # Afficher toutes les images
    plt.tight_layout()
    plt.show()

# Fonction pour appliquer toutes les transformations possibles à l'image
def apply_transformations(img_array):
    """
    Applique une variété de transformations sur l'image et renvoie une liste des images transformées.
    """
    # 1. Découpage de l'image (crop)
    cropped_img = img_array[20:180, 20:180]

    # 2. Miroir horizontal de l'image avec OpenCV (flip horizontal)
    mirrored_img = cv2.flip(img_array, 1)

    # 16. Flip vertical avec TensorFlow
    flipped_up_down_img = tf.image.flip_up_down(img_array)

    # 3. Blanchiment (standardisation)
    whitened_img = tf.image.per_image_standardization(img_array)

    # 4. Ajustement du contraste
    contrasted_img = tf.image.adjust_contrast(img_array, 2.0)

    # 5. Conversion en niveaux de gris et égalisation de l'histogramme
    gray_img = cv2.cvtColor(np.uint8(img_array), cv2.COLOR_RGB2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)

    # 6. Débruitage de l'image (suppression du bruit)
    denoised_img = cv2.fastNlMeansDenoisingColored(np.uint8(img_array), None, 10, 10, 7, 21)

    # 7. Redimensionnement de l'image
    resized_img = cv2.resize(img_array, (224, 224))

    # 8. Rotation de l'image (rotation de 45 degrés)
    center = (img_array.shape[1] // 2, img_array.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_img = cv2.warpAffine(img_array, matrix, (img_array.shape[1], img_array.shape[0]))

    # 9. Ajustement de la luminosité (brighter)
    brighter_img = tf.image.adjust_brightness(img_array, 0.3)

    # 10. Ajustement de la saturation
    saturated_img = tf.image.adjust_saturation(img_array, 2.0)

    # 11. Ajustement de la teinte
    hue_adjusted_img = tf.image.adjust_hue(img_array, 0.1)

    # 12. Zoom sur l'image (zoom)
    zoom_factor = 1.2
    zoomed_img = tf.image.central_crop(img_array, central_fraction=1/zoom_factor)

    # 13. Décalage horizontal et vertical (shift)
    # Ici, on décale beaucoup plus que précédemment pour appliquer un décalage significatif
    shifted_img = tf.image.pad_to_bounding_box(img_array, 50, 50, img_array.shape[0] + 100, img_array.shape[1] + 100)
    shifted_img = tf.image.crop_to_bounding_box(shifted_img, 50, 50, img_array.shape[0], img_array.shape[1])

    # 14. Application d'un filtre gaussien pour flouter l'image (blur)
    blurred_img = cv2.GaussianBlur(img_array, (7, 7), 0)

    # 15. Inversion des couleurs de l'image (inversion des canaux)
    inverted_img = cv2.bitwise_not(np.uint8(img_array))

    # Retourner toutes les images transformées
    images = [
        img_array, cropped_img, mirrored_img, flipped_up_down_img, whitened_img, contrasted_img, equalized_img, 
        denoised_img, resized_img, rotated_img, brighter_img, saturated_img, hue_adjusted_img, zoomed_img, 
        shifted_img, blurred_img, inverted_img
    ]
    titles = [
        'Original', 'Cropped', 'Mirrored (CV)', 'Flipped Up-Down (TF)', 'Whitened', 'Contrasted', 
        'Equalized', 'Denoised', 'Resized', 'Rotated', 'Brighter', 'Saturated', 'Hue Adjusted', 'Zoomed', 
        'Shifted', 'Blurred', 'Inverted'
    ]

    return images, titles


# Fonction pour afficher les images transformées
def display_images(images, titles):
    """
    Affiche une série d'images dans une grille 4x5 avec leurs titres respectifs.
    """
    plt.figure(figsize=(15, 12))  # Taille de la figure

    # Boucle pour afficher les images transformées
    for i, img in enumerate(images):
        plt.subplot(4, 5, i + 1)  # Positionnement dans la grille 4x5
        # Afficher l'image, avec conversion en uint8 si nécessaire et utilisation de cmap='gray' pour les images en niveaux de gris
        plt.imshow(np.uint8(img) if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None)
        plt.title(titles[i])  # Ajouter le titre correspondant à chaque transformation
        plt.axis('on')  
    plt.tight_layout()
    plt.show()  # Afficher toutes les images
    



########### Partie modèle personelle

# Data augmentation
def create_image_generator():
  return ImageDataGenerator(
      rotation_range=20,           # Rotation aléatoire jusqu'à 20 degrés
      width_shift_range=0.25,      # Décalage horizontal jusqu'à 25% de l'image
      height_shift_range=0.25,     # Décalage vertical jusqu'à 25% de l'image
      rescale=1./255,              # Normalisation des valeurs de pixels entre 0 et 1
      shear_range=0.25,            # Transformation de cisaillement jusqu'à 25%
      zoom_range=0.25,             # Zoom jusqu'à 25%
      horizontal_flip=True,        # Flip horizontal des images
      fill_mode='nearest',         # Remplissage des pixels manquants par la valeur la plus proche
      brightness_range=(0.9, 1.1), # Variation de luminosité entre 90% et 110% 
      channel_shift_range=0.1,     # Ajustement des canaux de couleur jusqu'à 10% 
      vertical_flip=True,          # flip vertical pour les images de chiens 
      validation_split=0.2         # Réservation de 20% des données pour la validation 
  )
  
  
  
# Création des générateurs pour les données d'entrainement et validation
def create_data_generators(img_generator, data_dir, target_size=(224, 224), batch_size=16):
  # Chargement des données d'entraînement
  train_generator = img_generator.flow_from_directory(
      data_dir,
      target_size=target_size,
      batch_size=batch_size,
      shuffle=True,
      class_mode='categorical',
      subset='training' 
  )

  # Chargement des données de validation
  validation_generator = img_generator.flow_from_directory(
      data_dir,
      target_size=target_size,
      batch_size=batch_size,
      shuffle=False,
      class_mode='categorical',
      subset='validation'
  )

  return train_generator, validation_generator


# Visualisation des images d'un batch
def visualize_batch(generator, num_rows=4, num_cols=8):
  # Obtenir le premier batch d'images et de cibles
  imgs, targets = next(iter(generator))

  # Créer une figure avec le nombre spécifié de lignes et colonnes
  fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*2.5, num_rows*2.5))

  # Parcourir les images du batch
  for i, (img, target) in enumerate(zip(imgs, targets)):
      if i >= num_rows * num_cols:
          break
      row = i // num_cols
      col = i % num_cols
      ax[row, col].imshow(img)
      class_index = np.argmax(target)
      class_name = list(generator.class_indices.keys())[class_index]
      ax[row, col].set_title(class_name, fontsize=8)
      ax[row, col].axis('off')  # Désactiver les axes pour une meilleure apparence

  plt.tight_layout()
  plt.show()


# Affichage de la matrice de confusion et du rapport de classification
def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """
    Affiche la matrice de confusion pour un modèle donné en utilisant des prédictions pré-calculées.

    Parameters:
    - y_true: Les vraies étiquettes des données de validation.
    - y_pred_classes: Les classes prédites par le modèle.
    - class_names: La liste des noms des classes.
    """
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    
    # Affichage de la heatmap pour la matrice de confusion
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g')
    ax.set_xlabel("Étiquettes prédites", color="b", fontsize=12)
    ax.set_ylabel("Vraies étiquettes", color="b", fontsize=12)
    ax.xaxis.set_ticklabels(class_names, rotation=45)
    ax.yaxis.set_ticklabels(class_names, rotation=45)
    plt.title("Matrice de Confusion", fontsize=16)
    plt.show()

# Affichage du rapport de classification
def classification_report_df(y_true, y_pred_classes, class_names):
    """
    Retourne le rapport de classification sous forme de DataFrame pour un modèle donné.

    Parameters:
    - y_true: Les vraies étiquettes des données de validation.
    - y_pred_classes: Les classes prédites par le modèle.
    - class_names: La liste des noms des classes.
    
    Returns:
    - df_report: Le rapport de classification sous forme de DataFrame arrondi à 2 chiffres.
    """
    
    # Calcul et formatage du rapport de classification
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # Arrondir les valeurs à 2 chiffres après la virgule
    df_report = df_report.round(2)

    # Retourner le DataFrame
    return df_report





# Création du graphique des courbes de perte
def plot_loss_curves(history, model_name="Model", color_chart=None):
  if color_chart is None:
      color_chart = ["#1f77b4",  # Bleu
               "#ff7f0e",  # Orange
               "#2ca02c",  # Vert
               "#d62728",  # Rouge
               "#9467bd",  # Violet
               "#8c564b",  # Marron
               "#e377c2",  # Rose
               "#7f7f7f"]  # Gris

  plt.figure(figsize=(10, 6))
  plt.plot(history.history["loss"], color=color_chart[0], label=f"Training loss ({model_name})")
  plt.plot(history.history["val_loss"], color=color_chart[1], label=f"Validation loss ({model_name})")
  plt.title(f'Training and Validation Loss - {model_name}')
  plt.xlabel('Epochs')
  plt.ylabel('Cross Entropy Loss')
  plt.legend()
  plt.grid(True)
  plt.show()
  
# Création du graphique des courbes d'accuracy
def plot_accuracy_curves(history, model_name="Model", color_chart=None):
    if color_chart is None:
        color_chart = ["#EF476F",  # Rose vif
               "#06D6A0",  # Vert menthe
               "#118AB2",  # Bleu océan
               "#FFD166",  # Jaune
               "#073B4C",  # Bleu marine
               "#8D99AE",  # Gris bleu
               "#EDF2F4",  # Blanc cassé
               "#FFBC42"]  # Orange vif
        
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], color=color_chart[2], label=f"Training Accuracy ({model_name})")
    plt.plot(history.history["val_accuracy"], color=color_chart[3], label=f"Validation Accuracy ({model_name})")
    plt.title(f'Training and Validation Accuracy - {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# Création du modèle simple
def create_simple_model(input_shape=(224, 224, 3), num_classes=3):
  model = Sequential([
      # Bloc 1 : Convolution + MaxPooling
      Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
      MaxPooling2D((2, 2)),  # Réduction de la dimension spatiale

      # Bloc 2 : Convolution + MaxPooling
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),  # Réduction de la dimension spatiale

      # Bloc 3 : Convolution + MaxPooling
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),  # Réduction de la dimension spatiale

      # Bloc 4 : Couches Fully Connected
      Flatten(),  # Aplatissement des données pour préparer l'entrée des couches fully-connected
      Dense(512, activation='relu'),  # Couche dense entièrement connectée avec 512 unités
      Dense(num_classes, activation='softmax')  # Couche de sortie avec activation softmax
  ])
  
  return model

# Création du modèle complexe
def create_complex_model(input_shape=(224, 224, 3), num_classes=3, l2_rate=0.0001):
    model = Sequential([
        # Bloc 1 : Convolution + BatchNorm + MaxPooling + Dropout
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        # Dropout(0.1),  

        # Bloc 2 : Convolution + BatchNorm + MaxPooling + Dropout
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.1),  

        # Bloc 3 : Convolution + BatchNorm + MaxPooling + Dropout
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        # Dropout(0.1),  

        # Bloc 4 : Convolution + BatchNorm + MaxPooling + Dropout
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        # Dropout(0.1),  

        # Couches de sortie
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(l2_rate)),
        Dropout(0.1),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_rate))
    ])
    
    return model



##### Modèles de transfert learning


# Fonction pour créer le modèle VGG16
def create_vgg_model(input_shape=(224, 224, 3), num_classes=3, dropout_rate=0.2):
    # Charger le modèle VGG16 pré-entraîné sans les couches fully connected
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Geler les couches du modèle de base
    base_model_vgg.trainable = False
    
    # Créer un nouveau modèle séquentiel
    model = Sequential([
        # Ajouter le modèle VGG16 de base
        base_model_vgg,
        
        # Ajouter les nouvelles couches fully connected
        GlobalAveragePooling2D(),
        Dense(512),
        BatchNormalization(),
        Activation('relu'),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model



# Fonction pour créer le modèle Xception
def create_xception_model(input_shape=(299, 299, 3), num_classes=3, dropout_rate=0.1):
    # Charger le modèle Xception pré-entraîné sans les couches fully connected
    base_model_xception = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Geler les couches du modèle de base
    base_model_xception.trainable = False
    
    # Créer un nouveau modèle séquentiel
    model = Sequential([
        # Ajouter le modèle Xception de base
        base_model_xception,
        
        # Ajouter les nouvelles couches fully connected
        GlobalAveragePooling2D(),  # Pooling global pour réduire la dimensionnalité
        Dense(512),  # Couche Dense avec 512 unités
        BatchNormalization(),  # Normalisation pour stabiliser l'apprentissage
        Activation('relu'),  # Activation ReLU
        Dropout(dropout_rate),  # Dropout pour éviter le surapprentissage
        Dense(num_classes, activation='softmax')  # Couche de sortie avec Softmax pour la classification
    ])
    
    return model


# Ajouter le callback ModelCheckpoint pour sauvegarder le modèle
def get_model_checkpoint(filepath='model/xception_best_model.keras'):
    checkpoint = ModelCheckpoint(filepath=filepath, 
                                 monitor='val_loss',  # Sauvegarder le modèle basé sur la meilleure validation loss
                                 verbose=1,
                                 save_best_only=True,  # Sauvegarder seulement le meilleur modèle
                                 mode='min')
    return checkpoint



