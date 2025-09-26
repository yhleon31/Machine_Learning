import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import zscore
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Ruta fija del archivo CSV
RUTA_DATASET = r'C:\Users\Equipo\Documents\MACHINE LEARNIG\Arbol de decisiones\spam_dataset.csv'

def ejecutar_modelo():
    try:
        df = pd.read_csv(RUTA_DATASET)

        # Verificar si la columna 'spam' existe
        if 'spam' not in df.columns:
            messagebox.showerror("Error", "La columna 'spam' no está en el dataset.")
            return

        # Renombrar la columna 'spam' como 'label' para consistencia
        df['label'] = df['spam']

        # Separar características y etiquetas
        X = df.drop(['spam', 'label'], axis=1)
        y = df['label']

        # Filtrar solo columnas numéricas
        X = X.select_dtypes(include=[np.number])
        if X.empty:
            messagebox.showerror("Error", "No hay columnas numéricas para entrenar el modelo.")
            return

        resultados = {'accuracy': [], 'f1': [], 'z': []}
        for seed in range(50):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
            model = DecisionTreeClassifier(criterion='gini', random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            resultados['accuracy'].append(accuracy_score(y_test, y_pred))
            resultados['f1'].append(f1_score(y_test, y_pred))
            resultados['z'].append(np.mean(np.abs(zscore(y_pred))))

        resultados_df = pd.DataFrame(resultados)
        mostrar_grafico(resultados_df)

        # 📊 Explicación automática de resultados
        acc_mean = np.mean(resultados['accuracy'])
        f1_mean = np.mean(resultados['f1'])
        z_mean = np.mean(resultados['z'])

        explicacion = f"""
🔍 Evaluación del clasificador CART basada en 50 iteraciones:

• Exactitud promedio: {acc_mean:.4f}
  → El modelo acertó en aproximadamente el {acc_mean*100:.2f}% de los casos. Esto refleja su capacidad para distinguir entre correos SPAM y HAM.

• F1 Score promedio: {f1_mean:.4f}
  → Mide el equilibrio entre precisión y exhaustividad. Es especialmente útil cuando las clases están desbalanceadas.

• Z Score promedio: {z_mean:.4f}
  → Indica la variabilidad de las predicciones entre ejecuciones. Un valor bajo sugiere que el modelo es estable.

📌 Interpretación:
Este análisis se realizó ejecutando el modelo 50 veces con diferentes divisiones de entrenamiento y prueba. Si la exactitud y el F1 Score son altos y el Z Score es bajo, el modelo es confiable y consistente. Si hay variaciones significativas, podrían deberse a la sensibilidad del árbol de decisión frente a los datos de entrada.
"""

        messagebox.showinfo("Explicación de resultados", explicacion)

    except Exception as e:
        messagebox.showerror("Error al ejecutar", str(e))

def mostrar_grafico(resultados_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=resultados_df['accuracy'], label='Accuracy', ax=ax, marker='o')
    sns.lineplot(data=resultados_df['f1'], label='F1 Score', ax=ax, marker='s')
    sns.lineplot(data=resultados_df['z'], label='Z Score', ax=ax, marker='^')
    ax.set_title('Desempeño del clasificador CART en 50 ejecuciones')
    ax.set_xlabel('Iteración')
    ax.set_ylabel('Valor de la métrica')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=ventana)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Clasificador SPAM/HAM - Árbol de Decisión CART")
ventana.geometry("800x600")

# Botón para ejecutar el modelo
boton_ejecutar = tk.Button(ventana, text="Ejecutar Clasificador", command=ejecutar_modelo, font=("Arial", 14), bg="#4CAF50", fg="white")
boton_ejecutar.pack(pady=20)

# Iniciar la GUI
ventana.mainloop()