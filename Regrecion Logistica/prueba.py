import os

ruta = r"C:\Users\Equipo\Documents\MACHINE LEARNIG\Regrecion Logistica\spam_dataset.csv"

if os.path.exists(ruta):
    print("✅ El archivo existe, podemos leerlo.")
else:
    print("❌ El archivo no se encuentra en la ruta dada.")
