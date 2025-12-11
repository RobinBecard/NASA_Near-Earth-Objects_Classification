# NASA Near-Earth Objects Classification Project

## Vue d'ensemble

Ce projet est un travail pratique du cours **IFT712 - Techniques d'apprentissage** à l'Université de Sherbrooke. L'objectif principal est de mettre en application les concepts fondamentaux de l'apprentissage automatique en implémentant et comparant six méthodes de classification distinctes sur un jeu de données réel provenant de la NASA.

## Objectifs

- Implémenter six algorithmes de classification différents
- Évaluer et comparer les performances de chaque modèle
- Analyser les forces et les faiblesses de chaque approche
- Fournir une analyse comparative détaillée basée sur des métriques standards

## Jeu de données

### Description

Le projet utilise le jeu de données **« NASA - Nearest Earth Objects »** disponible sur [Kaggle](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects). Ce jeu de données compile des informations relatives aux **objets proches de la Terre (NEO - Near-Earth Objects)** certifiés par la NASA, comprenant plus de **300 000 enregistrements** concernant les astéroïdes et comètes détectés par les systèmes d'observation de la NASA.

### Structure et Contenu

Le jeu de données contient :

- **Paramètres orbitaux** : distances relatives par rapport à la Terre, vitesses orbitales et vélocités absolues
- **Caractéristiques physiques** : diamètres estimés, albédos et propriétés de surface
- **Approches rapprochées** : résumés des approches et dates des événements
- **Évaluation des risques** : probabilités d'impact et classifications de danger
- **Métadonnées de découverte** : dates de découverte et observatoires ayant effectué les observations

### Sources et Instruments

Les données proviennent de sources institutionnelles majeures :

#### Sources principales
- **Centre for Near-Earth Object Studies (CNEOS)** - JPL/NASA
- **Minor Planet Center (MPC)** - Union Astronomique Internationale

#### Instruments de détection
- **Catalina Sky Survey (CSS)** - Trois télescopes (Mount Lemmon, Catalina, Siding Spring)
- **Spacewatch Program** - Kitt Peak Observatory
- **WISE/NEOWISE** - Télescope spatial infrarouge
- **Pan-STARRS** - Système de sondage panoramique
- **LONEOS** - Lowell Observatory Near-Earth Object Search

## Méthodes de Classification

Le projet implémente les six méthodes suivantes :

1. **Support Vector Machines (SVM)**
2. **Random Forest**
3. **Réseaux de Neurones**
4. **K-Nearest Neighbors (KNN)**
5. **Gradient Boosting**
6. **Logistic Regression**

## Métriques d'Évaluation

Chaque modèle est évalué selon les métriques standards :

- **Précision** : proportion de prédictions positives correctes
- **Rappel** : proportion de vrais positifs identifiés
- **F1-Score** : moyenne harmonique de la précision et du rappel
- **Area Under ROC Curve (AUC-ROC)** : performance globale du modèle
- **Matrice de confusion** : analyse détaillée des classifications

## Utilisation

### Prérequis

```
Python >= 3.8
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

### Installation

```bash
# Cloner le repository
git clone <https://github.com/RobinBecard/IFT712_Project.git>
cd IFT712_Project

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution

```bash
# Lancer le notebook Jupyter
jupyter notebook notebooks/IFT712_Project.ipynb
```

## Structure du Projet

```
IFT712_Project/
├── README.md                          # Documentation principale du projet
├── .gitignore                         # Configuration Git
├── requirements.txt                   # Dépendances Python
├── setup.py                           # Configuration du projet Python
├── IFT712_Project.pdf                 # Rapport final complet en pdf
│
├── src/                               # Code source principal
│   ├── __init__.py
│   ├── config.py                      # Configuration globale et paramètres
│   ├── data/                          # Module de gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py                  # Chargement des données
│   │   └── preprocessor.py            # Prétraitement et nettoyage
│   ├── models/                        # Module des modèles de classification
│   │   ├── __init__.py
│   │   ├── base_classifier.py         # Classe abstraite de base
│   │   ├── svm_classifier.py          # Implémentation SVM
│   │   ├── random_forest_classifier.py # Implémentation Random Forest
│   │   ├── neural_network_classifier.py # Implémentation Neural Network
│   │   ├── knn_classifier.py          # Implémentation KNN
│   │   ├── gradient_boosting_classifier.py # Implémentation Gradient Boosting
│   │   └── logistic_regression_classifier.py # Implémentation Logistic Regression
│   ├── evaluation/                    # Module d'évaluation et comparaison
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Calcul des métriques
│   │   └── comparison.py              # Comparaison des modèles
│   └── utils/                         # Utilitaires
│       ├── __init__.py
│       ├── visualization.py           # Génération de visualisations
│       └── utils.py                 # Fonctions utilitaires
│
├── notebooks/                         # Notebooks Jupyter
│   ├── IFT712_Project.ipynb            # Notebook principal du projet
│
├── datasets/                              # Répertoire de données
│   ├── neo.csv                      # Jeu de données principal
│
├── results/                           # Résultats et artefacts
│   ├── models/                        # Modèles entraînés (sérialisés)
│   ├── metrics/                       # Fichiers de métriques (CSV, JSON)
│   ├── plots/                         # Visualisations générées
│   └── comparisons/                   # Analyses comparatives
│


```

## Contenu du Notebook

Le notebook est structuré selon les sections suivantes :

### 1. Contexte
Introduction au cours IFT712 et objectifs du projet final.

### 2. Jeu de données
- Présentation détaillée du jeu de données NASA NEO
- Description de la structure et du contenu
- Sources de données et instruments de détection
- Flux de travail de collecte et validation

### 3. Implémentation
- Chargement et prétraitement des données
- Exploration et visualisation des données
- Implémentation des six algorithmes de classification
- Entraînement et évaluation de chaque modèle
- Analyse comparative des résultats

### 4. Résultats
- Comparaison des performances
- Visualisations des résultats
- Analyse critique des force et faiblesses

### 5. Conclusions
- Synthèse des découvertes
- Recommandations pour les développements futurs

## Analyse Comparative

Une analyse détaillée des performances permet d'identifier :

- Les modèles les plus performants pour cette application
- Les compromis entre précision et rappel
- L'impact de la complexité du modèle sur les performances
- Les cas d'usage optimaux pour chaque approche

## Ressources

- [Kaggle Dataset](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)
- [NASA CNEOS](https://cneos.jpl.nasa.gov/)
- [Minor Planet Center](https://www.minorplanetcenter.net/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## Licence

Ce projet est développé à titre éducatif pour le cours IFT712 de l'Université de Sherbrooke.

## Auteurs

Réalisé dans le cadre du programme de Master en Intelligence Artificielle et Science des Données à l'Université de Sherbrooke.

---

**Année académique** : 2025-2026  
**Cours** : IFT712 - Techniques d'apprentissage  
**Établissement** : Université de Sherbrooke