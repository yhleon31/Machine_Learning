# 📦 Importar librerías
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 🎯 1. Cargar dataset Iris
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")
target_names = iris.target_names

# 🔀 2. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 🧠 3. Modelo One-vs-Rest con regresión lineal
model = OneVsRestClassifier(LinearRegression())
model.fit(X_train, y_train)

# 📊 4. Predicciones (redondeamos y recortamos valores fuera de rango)
y_pred = np.round(model.predict(X_test)).astype(int)
y_pred = np.clip(y_pred, y.min(), y.max())

# ✅ 5. Evaluación del modelo
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

print("Classification report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_pred))

# 🎨 6. Visualizar la matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d",
            cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicción")
plt.ylabel("Verdadero")
plt.title(f"Matriz de Confusión - Accuracy: {acc:.4f}")
plt.tight_layout()
plt.show()
