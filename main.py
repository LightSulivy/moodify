import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Télécharger la derniere vesion du dataset
path = kagglehub.dataset_download("abdullahorzan/moodify-dataset")

print("Path to dataset files:", path)

# Trouver et le dataset csv
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

df = pd.DataFrame()
if csv_files:
    # Le charger en le convertissant en DataFrame
    csv_file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    print(f"Loaded dataset: {csv_files[0]}")
    print(df.columns)
else:
    print("No CSV file found in the downloaded dataset.")


# Separer les features et les valeurs objectifs
x = df.drop(columns=['labels',"Unnamed: 0"])
y = df["labels"]

print(x.head())
print(y.head())

# Diviser le dataset en donnee d'entrainement et donnee de test
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)



#Debut du pre-traitements des donnees (preprocessing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


# Note : Le résultat est un array Numpy. Pour le remettre en DataFrame (optionnel pour la visibilité) :
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=x.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=x.columns)

print(X_train_scaled_df.head())


def show_heatmap(df: pd.DataFrame):
    corr_matrix = df.corr()

    # 2. Création d'un masque pour cacher la moitié supérieure (redondante)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # 3. Configuration de la taille de la figure
    plt.figure(figsize=(10, 7))

    # 4. Affichage de la Heatmap
    _ = sns.heatmap(corr_matrix, 
                #mask=mask,            # Applique le masque triangulaire
                cmap='coolwarm',      # Palette de couleurs : Bleu (négatif) -> Rouge (positif)
                vmax=1,               # Valeur max de l'échelle
                center=0,             # Centre de l'échelle (blanc/neutre)
                square=True,          # Force les cellules à être carrées
                linewidths=.5,        # Lignes blanches entre les cases pour la lisibilité
                annot=True,           # Affiche les chiffres dans les cases
                fmt=".2f")            # Formate les chiffres (2 décimales)

    plt.title('Matrice de Corrélation des Features Audio', fontsize=15)
    plt.show()
    


show_heatmap(X_train_scaled_df)

# Suppression des features inutiles (qui ont trop de correlation avec d'autres)
features_a_supprimer = ['loudness']


# 2. Création du dataset propre
X_train_scaled_clean_df = X_train_scaled_df.drop(features_a_supprimer, axis=1)
X_test_scaled_clean_df = X_test_scaled_df.drop(features_a_supprimer, axis=1)