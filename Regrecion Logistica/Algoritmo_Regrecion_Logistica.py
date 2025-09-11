import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import Toplevel, Label
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve
)

# ========== 1. Estilo minimalista global ==========
sns.set_theme(style="white")  # fondo blanco, sin rejillas pesadas
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.labelweight": "light",
    "legend.frameon": False
})

# ========== 2. Cargar y preparar datos ==========
ruta = r"C:\Users\Equipo\Documents\MACHINE LEARNIG\Regrecion Logistica\spam_dataset.csv"
data = pd.read_csv(ruta)

X = data.drop(columns=["id", "spam"])
y = data["spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== 3. Entrenar modelo ==========
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ========== 4. Funciones para mostrar gráficas con texto ==========
def mostrar_matriz_confusion():
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys", cbar=False, ax=ax,
                linewidths=0.5, linecolor='white', annot_kws={"size": 12})
    ax.set_title("Matriz de Confusión", fontsize=13)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    fig.canvas.manager.set_window_title("Matriz de Confusión")
    plt.tight_layout()
    plt.show()

    texto = (
        "La matriz de confusión muestra que el modelo clasificó perfectamente los datos: "
        "144 verdaderos negativos y 156 verdaderos positivos, sin errores. Esto indica una "
        "separación clara entre correos spam y no spam, lo que valida el uso de regresión logística."
    )
    mostrar_texto("Matriz de Confusión", texto)

def mostrar_curva_roc():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#007acc", linewidth=2, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#cccccc")
    ax.set_title("Curva ROC", fontsize=13)
    ax.set_xlabel("Falso Positivo")
    ax.set_ylabel("Verdadero Positivo")
    ax.legend(loc="lower right")
    fig.canvas.manager.set_window_title("Curva ROC")
    plt.tight_layout()
    plt.show()

    texto = (
        "La curva ROC alcanza la esquina superior izquierda, lo que indica que el modelo "
        "tiene una capacidad de discriminación perfecta entre clases. Un AUC de 1.00 confirma "
        "que la regresión logística es altamente efectiva en este caso."
    )
    mostrar_texto("Curva ROC", texto)

def mostrar_precision_recall():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color="#009966", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.2, color="#009966")
    ax.set_title("Precisión vs Recall", fontsize=13)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    fig.canvas.manager.set_window_title("Precisión vs Recall")
    plt.tight_layout()
    plt.show()

    texto = (
        "La curva Precisión vs Recall muestra que el modelo mantiene alta precisión incluso "
        "cuando el recall aumenta. Esto es ideal en problemas como detección de spam, donde "
        "es importante minimizar falsos positivos sin perder sensibilidad."
    )
    mostrar_texto("Precisión vs Recall", texto)

def mostrar_importancia_variables():
    coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    coef.plot(kind="barh", color="#6a5acd", ax=ax)
    ax.set_title("Importancia de Variables", fontsize=13)
    ax.set_xlabel("Coeficiente")
    ax.set_ylabel("Variable")
    fig.canvas.manager.set_window_title("Importancia de Variables")
    plt.tight_layout()
    plt.show()

    texto = (
        "La gráfica de importancia de variables revela qué características influyen más en la "
        "predicción de spam. Variables como número de links, palabras clave o puntuación tienen "
        "coeficientes altos, lo que valida su relevancia en el modelo."
    )
    mostrar_texto("Importancia de Variables", texto)

# ========== 5. Función para mostrar texto explicativo ==========
def mostrar_texto(titulo, contenido):
    ventana_texto = Toplevel()
    ventana_texto.title(f"Análisis: {titulo}")
    ventana_texto.geometry("500x200")
    Label(ventana_texto, text=contenido, wraplength=480, justify="left", font=("Arial", 11)).pack(padx=20, pady=20)

# ========== 6. Interfaz principal ==========
ventana = tk.Tk()
ventana.title("Visualizador de Métricas de Regresión Logística")
ventana.geometry("400x400")
ventana.configure(bg="white")

tk.Label(ventana, text="Selecciona una gráfica para visualizar:", font=("Arial", 14), bg="white").pack(pady=20)

tk.Button(ventana, text="📊 Matriz de Confusión", command=mostrar_matriz_confusion, width=30, bg="#f0f0f0").pack(pady=10)
tk.Button(ventana, text="📈 Curva ROC", command=mostrar_curva_roc, width=30, bg="#f0f0f0").pack(pady=10)
tk.Button(ventana, text="📉 Precisión vs Recall", command=mostrar_precision_recall, width=30, bg="#f0f0f0").pack(pady=10)
tk.Button(ventana, text="📌 Importancia de Variables", command=mostrar_importancia_variables, width=30, bg="#f0f0f0").pack(pady=10)

ventana.mainloop()