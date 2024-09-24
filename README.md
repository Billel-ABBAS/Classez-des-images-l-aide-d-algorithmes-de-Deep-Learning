# Prédiction de la Race de Chiens 

## Introduction

Ce projet a été réalisé dans le cadre d'une étude approfondie sur les modèles de **computer vision** appliqués à la classification de la race des chiens. Il s'articule autour de la **comparaison entre différents modèles CNN** : des modèles personnalisés initialement construits, et l'utilisation de **modèles de transfert learning** préentraînés sur ImageNet.

L'approche adoptée est itérative, avec plusieurs étapes successives d'amélioration des performances des modèles :

1. **Modèle initial** : Entraînement d'un modèle CNN simple, avec du prétraitement d'images.
2. **Optimisations** : Ajout de techniques telles que le **Dropout** et la **BatchNormalization**.
3. **Data Augmentation** : Amélioration des performances avec des techniques de génération de données.
4. **Transfert Learning** : Utilisation des modèles préentraînés **VGG16** et **Xception** avec des couches personnalisées.

Les performances du modèle **Xception pré-entraîné** ont ensuite été comparées avec celles des modèles créés manuellement. Le dataset utilisé pour cet entraînement est constitué d'images de chiens triées en trois classes : **Silky Terrier**, **Golden Retriever** et **German Shepherd**.

Tous les entraînements ont été réalisés sur GPU à l'aide de Google Colab. Le meilleur modèle a été intégré dans un démonstrateur développé avec le framework **Streamlit**.

## Contenu du Repository

- **Notebooks d'entraînement des modèles** : Ils documentent les différentes étapes de l'entraînement et de l'amélioration des modèles.
  - **Notebook 1** : Prétraitement des données et augmentation d'images.
  - **Notebook 2** : Création et optimisation de modèles CNN personnalisés.
  - **Notebook 3** : Implémentation du transfert learning avec **VGG16** et **Xception**.
- **API Streamlit** : Application Web permettant de prédire la race d'un chien à partir d'une image fournie par l'utilisateur.
- **utils.py** : Fichier contenant les fonctions utilitaires pour le prétraitement des images, l'affichage des résultats et la création des modèles.

## Datasets Utilisés

Le dataset utilisé pour ce projet contient des images triées en trois classes de races de chiens :
- **n02097658-Silky_Terrier**
- **n02099601-Golden_Retriever**
- **n02106662-German_Shepherd**

Les données sont stockées dans le répertoire `data/`.

## Modèles et Démarche

### 1. Modèle Initial

Dans le **notebook 2**, un modèle **CNN simple** est créé à partir de zéro avec des couches de convolution et des couches fully connected. Ce modèle est amélioré progressivement avec l'ajout de techniques d'optimisation telles que le **Dropout** et la **BatchNormalization**.

### 2. Data Augmentation

La **data augmentation** est appliquée pour enrichir le dataset et améliorer la généralisation du modèle. Les transformations incluent des rotations, des flips horizontaux/verticaux, des ajustements de luminosité, du zoom, etc.

### 3. Transfert Learning

Le **transfert learning** est mis en œuvre dans le **notebook 3**, où les modèles **VGG16** et **Xception**, préentraînés sur ImageNet, sont utilisés comme base. Des couches personnalisées sont ajoutées pour adapter les modèles à la classification de trois classes de chiens.

### 4. Comparaison des Modèles

Les performances des modèles sont comparées à l'aide de :
- **Matrice de confusion** : Permet d'analyser les prédictions correctes et erronées pour chaque classe.
- **Courbes de perte** : Visualisation des pertes d'entraînement et de validation au fil des époques.
- **Rapport de classification** : Calcul de la précision, du rappel et du F1-score pour chaque classe.

## Résultats

Les résultats montrent que le modèle **Xception pré-entraîné** a surpassé les autres modèles, y compris le **VGG16** et les modèles personnalisés, en termes de précision globale et de stabilité des résultats.

## API de Prédiction avec Streamlit

Une application **Streamlit** a été développée pour permettre de prédire la race d'un chien à partir d'une image téléchargée par l'utilisateur. Le modèle **Xception** est utilisé pour effectuer la prédiction.

### Fonctionnalités de l'API :

- **Téléchargement d'image** : L'utilisateur peut télécharger une image de chien au format `.jpg`.
- **Redimensionnement automatique** : L'image est redimensionnée à 299x299 pixels, taille requise par Xception.
- **Prédiction** : Le modèle prédit la race du chien parmi les trois classes disponibles.
- **Affichage des résultats** : Le nom réel et le nom prédit de la race sont affichés sur l'interface.

### Lancer l'API

Pour lancer l'API avec **Streamlit**, exécutez la commande suivante :

```bash
streamlit run api.py
```

Ouvrez un navigateur à l'adresse : [http://localhost:8501](http://localhost:8501) pour utiliser l'application.

## Mode d'emploi

### Installation des dépendances

Assurez-vous d'avoir installé toutes les dépendances nécessaires. Vous pouvez les installer en exécutant :

```bash
pip install -r requirements.txt
```

### Structure du Projet

```
├── data                    # Dossiers contenant les images de chiens pour l'entraînement
│   ├── n02097658-silky_terrier
│   ├── n02099601-golden_retriever
│   └── n02106662-German_shepherd
├── model                   # Répertoire des modèles sauvegardés
│   └── xception_best_model.keras
├── Abbas_Billel_1_notebook_prétraitement_082024.ipynb
├── Abbas_Billel_2_notebook_model_perso_082024.ipynb
├── Abbas_Billel_3_notebook_model_transfer_learning_082024.ipynb
├── api.py                  # Fichier principal pour l'API Streamlit
├── utils.py                # Contient toutes les fonctions utilitaires (prétraitement, visualisation, modèles)
└── README.md               # Ce fichier
```

## Contribuer

Les contributions sont les bienvenues ! Si vous souhaitez contribuer à ce projet, suivez les étapes suivantes :
1. **Fork** le projet.
2. Créez une nouvelle branche (`git checkout -b feature/nouvelle-fonctionnalité`).
3. Faites vos modifications et commitez-les (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`).
4. Pushez sur la branche (`git push origin feature/nouvelle-fonctionnalité`).
5. Créez une **pull request**.
