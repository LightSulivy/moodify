# 🎵 Moodify - Classification de Mood Musical

Ce projet a pour objectif d'analyser et de classifier des morceaux de musique selon leur humeur ("Mood") en utilisant plusieurs modèles de Machine Learning.

Le dataset utilisé provient de Kaggle : [Moodify Dataset](https://www.kaggle.com/datasets/abdullahorzan/moodify-dataset).

## 📊 Fonctionnalités

Le script `main.py` effectue les étapes suivantes :

1.  **Téléchargement automatique** du dataset via `kagglehub`.
2.  **Préparation des données** :
    - Nettoyage et suppression des colonnes inutiles.
    - Analyse des outliers (Boxplots).
    - Standardisation des features (StandardScaler).
3.  **Visualisation** :
    - Matrice de corrélation (Heatmap) pour analyser les relations entre les features.
4.  **Comparaison de 4 Modèles de Machine Learning** :
    - 🟢 **Logistic Regression** (Baseline).
    - 🌲 **Random Forest Classifier** (Modèle ensembliste).
    - 📈 **Linear SVM** (Support Vector Machine optimisé).
    - 🚀 **HistGradientBoosting** (Gradient Boosting rapide pour grands datasets).
5.  **Évaluation** :
    - Calcul de la précision globale (Accuracy).
    - Rapport de classification détaillé (Precision, Recall, F1-score).
    - Matrices de confusion.
    - Analyse de l'importance des features (pour Random Forest).

## 🚀 Installation

Il est recommandé d'utiliser un environnement virtuel Python.

1.  **Cloner le dépôt :**

    ```bash
    git clone https://github.com/LightSulivy/moodify.git
    cd moodify
    ```

2.  **Créer un environnement virtuel (optionnel mais recommandé) :**

    ```bash
    python3 -m venv bin
    source bin/bin/activate  # Sur macOS/Linux
    # ou
    # bin\Scripts\activate  # Sur Windows
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Utilisation

Lancez simplement le script principal :

```bash
python3 main.py
```

Le script va télécharger les données, entraîner les modèles et afficher les résultats dans la console. Vous pouvez choisir d'activer ou non la Cross-Validation et le GridSearch via les invites interactives à la fin de l'exécution.

## 🏆 Résultats Comparatifs (Exemple)

Sur un jeu de données de ~278k musiques :

| Modèle                  | Précision (Accuracy) | Observations                                           |
| :---------------------- | :------------------- | :----------------------------------------------------- |
| **Gradient Boosting**   | **~96.3%** 🥇        | Meilleure performance globale.                         |
| **Random Forest**       | **~94.4%** 🥈        | Très robuste et performant.                            |
| **Logistic Regression** | ~84.0%               | Bon pour une baseline linéaire.                        |
| **SVM (Linear)**        | ~80.5%               | Moins adapté aux frontières de décision complexes ici. |

## 🛠 Technologies

- **Python 3.8+**
- **Pandas** (Manipulation de données)
- **Seaborn / Matplotlib** (Visualisation)
- **Scikit-Learn** (Machine Learning)
