# NASA Near-Earth Objects Classification Project

> Projet du cours **IFT712 - Techniques d'apprentissage** | Université de Sherbrooke | Session Automne 2025

## Vue d'ensemble

Implémentation et comparaison de **6 algorithmes de classification** sur le jeu de données [NASA Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) contenant plus de **300 000 enregistrements** d'astéroïdes et comètes.

## Algorithmes Implémentés

1. **SVM** (Support Vector Machine)
2. **Arbre de Décision**
3. **Réseaux de Neurones**
4. **Moindres Carrés**
5. **Bayes Naïf**
6. **Régression Logistique**

## Métriques d'Évaluation

Précision • Rappel • F1-Score • AUC-ROC • Matrice de confusion

## Installation & Utilisation

### Prérequis
```bash
Python >= 3.8
```

### Installation
```bash
git clone https://github.com/RobinBecard/IFT712_Project.git
cd IFT712_Project
pip install -r requirements.txt
```

### Exécution
```bash
jupyter notebook notebooks/IFT712_Project.ipynb
```

### Configuration
Le fichier `config.yaml` centralise tous les paramètres du projet.

```python
from src.config import get_config
config = get_config()
dataset_path = config.get_path('paths.dataset')
```

## Structure du Projet

```
IFT712_Project/
├── README.md
├── config.yaml                    # Configuration centralisée
├── requirements.txt
├── IFT712_Project.pdf            # Rapport final
├── src/
│   ├── config.py                 # Gestionnaire de configuration
│   ├── data/                     # Chargement et prétraitement
│   ├── models/                   # 6 algorithmes de classification
│   └── utils/                    # Visualisations et utilitaires
├── notebooks/
│   └── IFT712_Project.ipynb      # Notebook principal
├── datasets/
│   └── neo.csv                   # Données NASA NEO
└── results/                      # Modèles, métriques, graphiques
```

## Contenu du Notebook

1. **Contexte** - Introduction et objectifs
2. **Jeu de données** - Exploration et analyse des données NASA NEO
3. **Implémentation** - Développement et entraînement des 6 modèles
4. **Résultats** - Comparaison des performances et visualisations
5. **Conclusions** - Synthèse et recommandations

## Ressources

- [Dataset Kaggle](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)
- [NASA CNEOS](https://cneos.jpl.nasa.gov/)
- [Minor Planet Center](https://www.minorplanetcenter.net/)

---

**Année académique** : 2025-2026  
**Cours** : IFT712 - Techniques d'apprentissage  
**Établissement** : Université de Sherbrooke