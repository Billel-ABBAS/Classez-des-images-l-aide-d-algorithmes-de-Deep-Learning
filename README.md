# Projet de Classification de Races de Chiens avec CNN et Transfer Learning

Ce projet a été réalisé dans le cadre du cursus d'ingénieur machine learning proposé par centralesupélec et Openclassrooms.

L'objectif est de comparer un modèle CNN développé "from scratch" avec des modèles pré-entraînés utilisant le **Transfer Learning**. J'ai exploré plusieurs approches, notamment les architectures **VGG16** et **Xception**, pour classifier des races de chiens.

## Résultats

Les résultats montrent que le modèle **Xception pré-entraîné** a surpassé les autres modèles, y compris le **VGG16** et les modèles personnalisés, en termes de précision globale et de stabilité.

## API de Prédiction avec Streamlit

Une application **Streamlit** a été développée pour permettre de prédire la race d'un chien à partir d'une image téléchargée. Le modèle **Xception** est utilisé pour cette tâche.

### Fonctionnalités de l'API :

- **Téléchargement d'image** : L'utilisateur peut télécharger une image de chien au format `.jpg`.
- **Redimensionnement automatique** : L'image est redimensionnée à 299x299 pixels, taille requise par Xception.
- **Prédiction** : Le modèle prédit la race du chien parmi les trois classes disponibles.
- **Affichage des résultats** : Le nom réel et le nom prédit de la race sont affichés.

### Déploiement de l'API

L'application **Streamlit** est déployée sur **Streamlit Cloud** et accessible via ce lien :

[Accéder à l'application Streamlit](https://classez-des-images-l-aide-d-algorithmes-de-deep-learning-giqxy.streamlit.app/)

---

## Mode d'emploi

### Installation des dépendances

Pour installer les dépendances, exécutez :

```bash
pip install -r requirements.txt
```

---

## Structure du Projet et Contenu des Notebooks

Le projet est structuré de manière à inclure les fichiers essentiels pour la construction, l'entraînement et l'évaluation des modèles de classification de races de chiens.

```
├── data                    # Dossiers contenant les images de chiens pour l'entraînement
│   ├── n02097658-silky_terrier
│   ├── n02099601-golden_retriever
│   └── n02106662-German_shepherd
├── model                   # Répertoire des modèles sauvegardés
│   └── xception_best_model.keras
├── Abbas_Billel_1_notebook_prétraitement_082024.ipynb   # Notebook 1 : Prétraitement des images
├── Abbas_Billel_2_notebook_model_perso_082024.ipynb     # Notebook 2 : Création du modèle personnalisé CNN
├── Abbas_Billel_3_notebook_model_transfer_learning_082024.ipynb  # Notebook 3 : Modèles pré-entraînés (VGG16 et Xception)
├── api.py                  # Fichier principal pour l'API Streamlit
├── utils.py                # Contient toutes les fonctions utilitaires (prétraitement, visualisation, modèles)
└── README.md               # Ce fichier
```

### Contenu des Notebooks

1. **Notebook 1 - Prétraitement des images :**
   - Chargement et transformation des images de chiens (redimensionnement, ajustements de contraste, saturation, etc.).
   - Application de techniques de **data augmentation** pour améliorer la robustesse des modèles.
   - Visualisation des images transformées avec des exemples de techniques telles que le **flipping**, le **cropping**, et la **normalisation**.

2. **Notebook 2 - Création du modèle personnalisé CNN :**
   - **Modèle simple** :
     - Construction d'un **CNN basique** avec trois blocs de convolution suivis de couches de **max-pooling** et d'une couche entièrement connectée.
     - Ce modèle a montré des performances de base, mais avec des limitations en termes de précision et de généralisation.
   - **Modèle complexe** :
     - Amélioration du modèle simple avec davantage de couches de **convolution**, de **BatchNormalization**, et de **Dropout** pour régulariser l'apprentissage et éviter le surapprentissage.
     - Ce modèle a montré des performances améliorées, mais n'a pas surpassé les modèles pré-entraînés en termes de précision et de stabilité.

3. **Notebook 3 - Modèles pré-entraînés (VGG16 et Xception) :**
   - Utilisation de modèles de **transfer learning** comme **VGG16** et **Xception**, pré-entraînés sur ImageNet.
   - Comparaison des performances de ces modèles avec le modèle CNN personnalisé.
   - Évaluation finale des modèles avec **Xception** comme modèle le plus performant pour la classification des races de chiens.




