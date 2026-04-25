# NASA Near-Earth Objects - Classification

> Projet du cours **IFT712 - Techniques d'apprentissage** | Université de Sherbrooke | Automne 2025
> Équipe : Adrien SKRZYPCZAK · Cédric HAN · Robin BECARD

[![Python](https://img.shields.io/badge/Python-3.1O+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Dataset](https://img.shields.io/badge/Kaggle-NASA%20NEO-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)

## Présentation

Ce projet implémente et compare **6 algorithmes de classification scikit-learn** sur le jeu de données NASA Nearest Earth Objects (90 836 astéroïdes, dont ~9,7 % dangereux). L'objectif est de prédire la dangerosité d'un objet céleste en respectant les meilleures pratiques de validation croisée, de recherche d'hyperparamètres et de gestion du déséquilibre de classes.

## Algorithmes Implémentés

| Algorithme | Classe | Approche |
|---|---|---|
| Arbre de Décision | `DecisionTreeModel` | Hiérarchique, interprétable |
| SVM | `SVMClassifier` | Non-linéaire, marges maximales |
| Régression Logistique | `LogisticRegressionClassifier` | Linéaire, baseline |
| Bayes Naïf | `NaiveBayesClassifier` | Probabiliste |
| Réseau de Neurones (MLP) | `NNClassifier` | Non-linéaire, profond |
| Moindres Carrés (Ridge) | `LeastSquaresClassifier` | Linéaire, baseline |

## Résultats clés

| Modèle | Accuracy | Recall | F1-Score | AUC-ROC | Temps (s) |
|---|---|---|---|---|---|
| Least Squares | 73.71% | **98.98%** | 42.49% | - | **0.01** |
| SVM | 77.22% | **99.43%** | 46.71% | 0.902 | 1026.8 |
| Régression Logistique | 78.49% | 95.02% | **46.23%** | 0.889 | 0.31 |
| Arbre de Décision | 91.43% | 13.57% | 23.56% | 0.916 | 0.12 |
| Réseau de Neurones | **91.45%** | 15.10% | 29.16% | **0.915** | 9.05 |
| Bayes Naïf | 83.23% | 46.61% | 35.10% | 0.864 | 0.02 |

> **Choix de métrique** : dans un contexte de défense planétaire, le **Recall** est la métrique prioritaire - manquer un astéroïde dangereux a des conséquences catastrophiques.

## Contenu du Notebook

1. **Contexte** - Introduction et objectifs
2. **Jeu de données** - Exploration et analyse des données NASA NEO
3. **Implémentation** - Développement et entraînement des 6 modèles
4. **Résultats** - Comparaison des performances et visualisations
5. **Conclusions** - Synthèse et recommandations

## Pipeline de prétraitement

1. **Suppression** des colonnes invariantes (`orbiting_body`, `sentry_object`) et identifiants (`id`, `name`)
2. **Transformation logarithmique** (`np.log1p`) sur `est_diameter_max`, `relative_velocity`, `miss_distance`
3. **RobustScaler** pour limiter l'impact des outliers
4. **Split stratifié 80/20** pour préserver le ratio de classes (~9.73 % dangereux)
5. **Pondération des classes** (`class_weight='balanced'`) sur les modèles linéaires et SVM

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
Tous les paramètres (chemins, hyperparamètres, seeds) sont centralisés dans `config.yaml` :

```python
from src.config import get_config
config = get_config()
dataset_path = config.get_path('paths.dataset')
```

### Optimisation des hyperparamètres

```bash
python src/optimize_all_models.py
# Interface interactive pour sélectionner les modèles à optimiser
# Les meilleurs hyperparamètres sont exportés au format YAML
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

## Ressources

- [Dataset Kaggle - NASA Nearest Earth Objects](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects)
- [NASA CNEOS](https://cneos.jpl.nasa.gov/)
- [Minor Planet Center](https://www.minorplanetcenter.net/)
- [Rapport du projet (PDF)](./IFT712_Project.pdf)

---

**Cours** : IFT712 - Techniques d'apprentissage · **Université de Sherbrooke** · Automne 2025
