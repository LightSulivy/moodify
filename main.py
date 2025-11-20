import kagglehub
import pandas as pd
import numpy as np
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

print(X_train_scaled_df.head())