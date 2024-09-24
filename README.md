# Projet de Classification de Races de Chiens avec CNN et Transfer Learning

Le projet de ce notebook a été réalisé dans le cadre du cursus d'ingénieur machine learning proposé par Openclassrooms.

L'objectif du projet est de comparer un modèle CNN développé from "Scratch" avec des modèles CNN pré-entraînés utilisant du **Transfer Learning**. Plusieurs approches ont été testées pour entraîner un modèle capable de classifier des races de chiens, notamment les architectures **VGG16** et **Xception**.

## Résultats

Les résultats montrent que le modèle **Xception pré-entraîné** a surpassé les autres modèles, y compris le **VGG16** et les modèles personnalisés, en termes de précision globale et de stabilité des résultats.

## API de Prédiction avec Streamlit

Une application **Streamlit** a été développée pour permettre de prédire la race d'un chien à partir d'une image téléchargée par l'utilisateur. Le modèle **Xception** est utilisé pour effectuer la prédiction.

### Fonctionnalités de l'API :

- **Téléchargement d'image** : L'utilisateur peut télécharger une image de chien au format `.jpg`.
- **Redimensionnement automatique** : L'image est redimensionnée à 299x299 pixels, taille requise par Xception.
- **Prédiction** : Le modèle prédit la race du chien parmi les trois classes disponibles.
- **Affichage des résultats** : Le nom réel et le nom prédit de la race sont affichés sur l'interface.

### Déploiement de l'API

L'application **Streamlit** a été déployée sur **Streamlit Cloud** et peut être utilisée directement à partir de ce lien :

[Accéder à l'application Streamlit](https://geekderbzsqrebfdqdbnce.streamlit.app/)

Vous pouvez utiliser cette application pour télécharger une image de chien et obtenir une prédiction de la race de chien parmi les trois classes disponibles.

---

## Mode d'emploi

### Installation des dépendances

Assurez-vous d'avoir installé toutes les dépendances nécessaires. Vous pouvez les installer en exécutant :

```bash
pip install -r requirements.txt
```

---

## Structure du Projet et Contenu des Notebooks

Le projet contient plusieurs fichiers essentiels à la construction, l'entraînement et l'évaluation des modèles de classification de races de chiens.

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
   - Visualisation des images transformées avec des exemples de techniques telles que le **flipping**, le **cropping** et la **normalisation**.

2. **Notebook 2 - Création du modèle personnalisé CNN :**
   - **Modèle simple** :
     - Construction d'un **CNN basique** avec trois blocs de convolution suivis de couches de **max-pooling** et d'une couche entièrement connectée.
     - Ce modèle a montré des performances de base mais a été limité en termes de précision et de généralisation sur les données de validation.
   - **Modèle complexe** :
     - Amélioration du modèle simple en ajoutant plus de couches de **convolution**, de **BatchNormalization**, et de **Dropout** pour mieux régulariser l'apprentissage et prévenir le surapprentissage.
     - Bien que ce modèle soit plus performant que le modèle simple, il n'a pas atteint les niveaux de précision et de stabilité obtenus avec les modèles pré-entraînés.

3. **Notebook 3 - Modèles pré-entraînés (VGG16 et Xception) :**
   - Utilisation de modèles de **transfer learning** pré-entraînés tels que **VGG16** et **Xception**, initialement entraînés sur ImageNet.
   - Comparaison des performances de ces modèles pré-entraînés avec le modèle personnalisé CNN.
   - Évaluation finale et choix du modèle **Xception** comme meilleur modèle pour la tâche de classification de races de chiens.


