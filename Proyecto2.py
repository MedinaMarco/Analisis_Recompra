import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix 

# Intentar leer el Excel; si no existe, crear los datos y guardarlos primero
try:
    df = pd.read_excel("Mini_Proyecto_Clientes_Promociones.xlsx", engine="openpyxl")
except FileNotFoundError:
    data = {
        "Cliente_ID": range(1, 21),
        "Genero": ["F", "M"] * 10,
        "Edad": [23, 34, 45, 29, 31, 38, 27, 50, 40, 36, 25, 33, 46, 28, 39, 42, 30, 48, 35, 37],
        "Recibio_Promo": ["Sí", "No", "Sí", "Sí", "No", "Sí", "No", "Sí", "No", "Sí", "No", "Sí", "Sí", "No", "No", "Sí", "No", "Sí", "No", "Sí"],
        "Monto_Promocion": [500, 0, 700, 300, 0, 600, 0, 800, 0, 450, 0, 620, 710, 0, 0, 480, 0, 750, 0, 520],
        "Recompra": ["Sí", "No", "Sí", "No", "No", "Sí", "No", "Sí", "No", "Sí", "No", "No", "Sí", "No", "No", "Sí", "No", "Sí", "No", "Sí"],
        "Total_Compras": [2, 1, 3, 1, 1, 4, 1, 5, 1, 3, 1, 2, 4, 1, 1, 3, 1, 5, 1, 3],
        "Ingreso_Mensual": [30000, 45000, 40000, 28000, 32000, 50000, 31000, 60000, 29000, 37000, 31000, 34000, 47000, 30000, 29000, 43000, 33000, 55000, 30000, 41000]
    }
    df = pd.DataFrame(data)
    df.to_excel("Mini_Proyecto_Clientes_Promociones.xlsx", index=False)
    print("Archivo Excel generado correctamente.")

# Información básica del dataset
print("="*50)
print("INFORMACIÓN DEL DATASET")
print("="*50)
df.info() 
print("\n" + "="*50)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*50)
print(df.describe())

# Preprocesamiento de datos
df['Genero'] = df['Genero'].map({'F': 0, 'M': 1}) 
df['Recibio_Promo'] = df['Recibio_Promo'].map({'Sí': 1, 'No': 0}) 
df['Recompra'] = df['Recompra'].map({'Sí': 1, 'No': 0})

# Visualizaciones
plt.figure(figsize=(15, 5))

# GRÁFICO 1: Recompra según el Monto Promocional
plt.subplot(1, 3, 1)
sns.boxplot(x="Recompra", y="Monto_Promocion", data=df) 
plt.title("Recompra según el Monto Promocional")
plt.xlabel("Recompra (0=No, 1=Sí)")
plt.ylabel("Monto Promoción")

# GRÁFICO 2: Distribución de Edad por Recompra
plt.subplot(1, 3, 2)
sns.histplot(data=df, x='Edad', hue='Recompra', kde=True, alpha=0.6)
plt.title("Distribución de Edad por Recompra")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")

# GRÁFICO 3: Total de Compras vs Recompra
plt.subplot(1, 3, 3)
sns.boxplot(x="Recompra", y="Total_Compras", data=df)
plt.title("Total de Compras vs Recompra")
plt.xlabel("Recompra (0=No, 1=Sí)")
plt.ylabel("Total Compras")

plt.tight_layout()
plt.show()

# Entrenamiento del modelo
X = df.drop(['Cliente_ID', 'Recompra'], axis=1) 
y = df['Recompra'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)  # Limitamos la profundidad para mejor visualización
modelo.fit(X_train, y_train) 
y_pred = modelo.predict(X_test) 

# Métricas del modelo
print("\n" + "="*50)
print("EVALUACIÓN DEL MODELO")
print("="*50)
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred)) 
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Importancia de las características
print("\n" + "="*50)
print("IMPORTANCIA DE LAS CARACTERÍSTICAS")
print("="*50)
importancias = modelo.feature_importances_
caracteristicas = X.columns
for caracteristica, importancia in zip(caracteristicas, importancias):
    print(f"{caracteristica}: {importancia:.4f}")

# Visualización del árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(modelo, 
          feature_names=X.columns,
          class_names=['No Recompra', 'Recompra'],
          filled=True,
          rounded=True,
          fontsize=12,
          proportion=True)

plt.title("Árbol de Decisión - Predicción de Recompra", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Visualización alternativa más compacta
plt.figure(figsize=(15, 8))
plot_tree(modelo, 
          feature_names=X.columns,
          class_names=['No', 'Sí'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # Mostramos solo 3 niveles para mejor legibilidad

plt.title("Árbol de Decisión - Predicción de Recompra (3 niveles)", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# Gráfico de importancia de características
plt.figure(figsize=(10, 6))
importancias_df = pd.DataFrame({
    'Caracteristica': caracteristicas,
    'Importancia': importancias
}).sort_values('Importancia', ascending=True)

plt.barh(importancias_df['Caracteristica'], importancias_df['Importancia'])
plt.xlabel('Importancia')
plt.title('Importancia de las Características en el Árbol de Decisión')
plt.tight_layout()
plt.show()

# Predicciones de ejemplo
print("\n" + "="*50)
print("PREDICCIONES DE EJEMPLO")
print("="*50)
print("Datos de prueba:")
print(X_test)
print("\nPredicciones vs Real:")
resultados = pd.DataFrame({
    'Real': y_test.values,
    'Predicción': y_pred
})
print(resultados)

# Análisis de las reglas del árbol
print("\n" + "="*50)
print("ANÁLISIS DE LAS REGLAS DEL ÁRBOL")
print("="*50)
print("El árbol de decisión toma decisiones basándose en:")

# Mostrar algunas reglas importantes
if importancias[0] > 0:  # Característica más importante
    print(f"- La característica más importante es: {caracteristicas[0]} (importancia: {importancias[0]:.3f})")
if importancias[1] > 0:  # Segunda característica más importante
    print(f"- La segunda característica más importante es: {caracteristicas[1]} (importancia: {importancias[1]:.3f})")

print("\nInterpretación del modelo:")
print("El árbol clasifica a los clientes en 'Recompra' o 'No Recompra' basándose en:")
print("1. Si recibieron promoción y el monto de la misma")
print("2. Su historial de compras totales")
print("3. Su edad e ingreso mensual")