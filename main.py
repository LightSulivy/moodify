import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

"""
1. Nettoyé les données (StandardScaler).

2. Analysé les corrélations (Heatmap) et fait des choix métier (spec_rate).

3. Établi une baseline (Logistic Regression ~84%).

4. Challenge avec un modèle complexe (Random Forest ~94.3%).

5. Optimisé le tout (GridSearchCV ~94.55%)

"""


# Télécharger la derniere vesion du dataset
path = kagglehub.dataset_download("abdullahorzan/moodify-dataset")

print("Path to dataset files:", path)

# Trouver et le dataset csv
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

if csv_files:
    # Le charger en le convertissant en DataFrame
    csv_file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    print(f"Loaded dataset: {csv_files[0]}")
    print(df.columns)
else:
    print("No CSV file found in the downloaded dataset.")


# Separer les features et les valeurs objectifs
cols_to_drop = ["labels", "Unnamed: 0", "Unnamed: 0.1", "uri"]
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
x = df.drop(columns=cols_to_drop)
y = df["labels"]

print(x.head())
print(y.head())


def analyze_outliers(df):
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df_numeric = df[numeric_cols]

    # Visualisation avec Boxplot
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df_numeric)
    plt.xticks(rotation=90)
    plt.title("Distribution des features et Outliers")
    # plt.show()

    # Comptage des outliers (Méthode IQR)
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).sum()

    print("\n--- Nombre d'outliers détectés par feature ---")
    print(outliers[outliers > 0])


analyze_outliers(x)


# Diviser le dataset en donnee d'entrainement et donnee de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Debut du pre-traitements des donnees (preprocessing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# Le résultat est un array Numpy. Pour le remettre en DataFrame (optionnel pour la visibilité) :
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=x.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=x.columns)

print(X_train_scaled_df.head())


def show_heatmap(df: pd.DataFrame):
    corr_matrix = df.corr()

    # Configuration de la taille de la figure
    _ = plt.figure(figsize=(10, 7))

    # Affichage de la Heatmap
    _ = sns.heatmap(
        corr_matrix,
        # mask=mask,            # Applique le masque triangulaire
        cmap="coolwarm",  # Palette de couleurs : Bleu (négatif) -> Rouge (positif)
        vmax=1,  # Valeur max de l'échelle
        center=0,  # Centre de l'échelle (blanc/neutre)
        square=True,  # Force les cellules à être carrées
        linewidths=0.5,  # Lignes blanches entre les cases pour la lisibilité
        annot=True,  # Affiche les chiffres dans les cases
        fmt=".2f",
    )  # Formate les chiffres (2 décimales)

    _ = plt.title("Matrice de Corrélation des Features Audio", fontsize=15)
    # plt.show()


show_heatmap(X_train_scaled_df)

# Suppression des features inutiles (qui ont trop de correlation avec d'autres)
features_a_supprimer = []

# Création du dataset propre
X_train_scaled_clean_df = X_train_scaled_df.drop(features_a_supprimer, axis=1)
X_test_scaled_clean_df = X_test_scaled_df.drop(features_a_supprimer, axis=1)

print("Features conservées :", X_train_scaled_clean_df.columns.tolist())

# Initialisation du modèle
# max_iter=1000 permet de laisser plus de temps au modèle pour trouver la solution mathématique
model_lr = LogisticRegression(random_state=42, max_iter=1000)
# n_estimators=100 : On crée 100 arbres de décision
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=1,
)
# Modèle SVM (Linear Support Vector Machine - plus rapide)
svm_model = LinearSVC(dual=False, random_state=42)

# Modèle Gradient Boosting (HistGradientBoosting - optimisé pour grands datasets)
gb_model = HistGradientBoostingClassifier(random_state=42)


# Entraînement (Fit) sur les données d'entraînement
print("Entraînement 1/4")
model_lr.fit(X_train_scaled_clean_df, y_train)
print("Entraînement 2/4")
rf_model.fit(X_train_scaled_clean_df, y_train)
print("Entraînement 3/4")
svm_model.fit(X_train_scaled_clean_df, y_train)
print("Entraînement 4/4")
gb_model.fit(X_train_scaled_clean_df, y_train)

# Prédiction sur les données de test (jamais vues par le modèle)
y_pred_lr = model_lr.predict(X_test_scaled_clean_df)
y_pred_rf = rf_model.predict(X_test_scaled_clean_df)
y_pred_svm = svm_model.predict(X_test_scaled_clean_df)
y_pred_gb = gb_model.predict(X_test_scaled_clean_df)


# Calcul du score global (Accuracy)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"✅ Précision globale LR : {accuracy_lr:.2%}")
print(f"✅ Précision globale RF : {accuracy_rf:.2%}")
print(f"✅ Précision globale SVM : {accuracy_svm:.2%}")
print(f"✅ Précision globale GB : {accuracy_gb:.2%}")


# Affiche les métriques détaillées par Mood
print("\n--- Rapport de Classification ---")
print(classification_report(y_test, y_pred_lr))
print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print("\n--- SVM ---")
print(classification_report(y_test, y_pred_svm))
print("\n--- Gradient Boosting ---")
print(classification_report(y_test, y_pred_gb))

# Visualisation de la Matrice de Confusion


def showHeatmapConfusion(y_test, y_pred, nameModel: str, model):
    _ = plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test, y_pred)

    _ = sns.heatmap(
        conf_matrix,
        annot=True,  # Affiche les nombres
        fmt="d",  # Format 'd' pour des entiers (pas de notation scientifique)
        cmap="Blues",  # Bleu pour rester lisible
        xticklabels=model.classes_,  # Noms des moods en bas
        yticklabels=model.classes_,
    )  # Noms des moods à gauche

    _ = plt.title("Matrice de Confusion " + nameModel)
    _ = plt.xlabel("Mood Prédit")
    _ = plt.ylabel("Vrai Mood")
    # plt.show()


showHeatmapConfusion(y_test, y_pred_lr, "Logistic Regression", model_lr)
showHeatmapConfusion(y_test, y_pred_rf, "Random Forest", rf_model)
showHeatmapConfusion(y_test, y_pred_svm, "SVM", svm_model)
showHeatmapConfusion(y_test, y_pred_gb, "Gradient Boosting", gb_model)


# Création d'un petit DataFrame pour lier le nom des colonnes à leur score d'importance
feature_importances = pd.DataFrame(
    {
        "feature": X_train_scaled_clean_df.columns,
        "importance": rf_model.feature_importances_,
    }
).sort_values(by="importance", ascending=False)

# Affichage graphique
_ = plt.figure(figsize=(10, 6))
_ = sns.barplot(
    x="importance",
    y="feature",
    data=feature_importances,
    hue="feature",
    legend=False,
    palette="viridis",
)
_ = plt.title("Quelles features définissent le plus le Mood ?")
_ = plt.xlabel("Importance (Poids dans la décision)")
_ = plt.ylabel("Features")
# plt.show()


user_input = (
    input("Voulez-vous lancer la cross validation ? (yes/no) : ").strip().lower()
)

if user_input == "yes":
    print("Début de la cross validation")
    # Concatenation des données pour la validation croisée
    X_full = pd.concat([X_train_scaled_clean_df, X_test_scaled_clean_df], axis=0)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_test)], axis=0)

    # Lancement de la validation croisée
    scores = cross_val_score(rf_model, X_full, y_full, cv=5)
    print(f"Scores des 5 tests : {scores}")
    print(f"✅ Moyenne réelle : {scores.mean():.2%} (+/- {scores.std():.2%})")
else:
    print("Validation croisée annulée.")


user_input = (
    input(
        "Voulez-vous lancer le grid search ? \n /!\\ Warning cela peut prendre plusieurs dizaines de minutes voir plusieurs heures\n(yes/no) : "
    )
    .strip()
    .lower()
)

if user_input == "yes":
    # 1. Définition de la grille de paramètres à tester
    # On teste des variations autour des valeurs par défaut
    param_grid = {
        "n_estimators": [
            100,
            200,
            300,
        ],  # Plus d'arbres = souvent mieux, mais plus lent
        "max_depth": [None, 15, 25],  # Limiter la profondeur évite le surapprentissage
        "min_samples_leaf": [1, 2, 4],  # Nombre min d'exemples pour valider une feuille
        "criterion": [
            "gini",
            "entropy",
        ],  # La formule mathématique pour diviser les nœuds
    }
    # 2. Configuration du GridSearch
    # cv=5 : On continue de faire de la validation croisée (5 tests par combinaison)
    # n_jobs=-1 : Utilise tous les cœurs de ton processeur pour aller plus vite
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring="accuracy",
    )

    # 3. Lancement de la recherche (peut prendre 1 à 3 minutes selon ton PC)
    print("🕵️ Recherche des meilleurs hyperparamètres en cours...")
    grid_search.fit(X_train_scaled_clean_df, y_train)

    # 4. Résultats
    print(f"\n🏆 Meilleur score trouvé : {grid_search.best_score_:.2%}")
    print("⚙️ Meilleurs paramètres :", grid_search.best_params_)

    # 5. Mise à jour de ton modèle avec le vainqueur
    best_rf = grid_search.best_estimator_

    # 🏆 Meilleur score trouvé : 94.42%
    # ⚙️ Meilleurs paramètres : {'criterion': 'entropy', 'max_depth': None, 'min_samples_leaf': 1, 'n_estimators': 300}
else:
    print("Cross validation annulée.")
