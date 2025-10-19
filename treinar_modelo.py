import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# === Carrega o dataset ===
dados = pd.read_csv("data/dataset_libras.csv", encoding="latin1")

# === Separa X e y ===
X = dados.drop(columns=["letra"])
y = dados["letra"]

# === Divide treino e teste ===
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# === Treina o modelo ===
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_treino, y_treino)

# === Avalia ===
acuracia = modelo.score(X_teste, y_teste)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# === Salva o modelo treinado ===
with open("models/modelo_libras.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("✅ Modelo salvo em 'modelo_libras.pkl'")
