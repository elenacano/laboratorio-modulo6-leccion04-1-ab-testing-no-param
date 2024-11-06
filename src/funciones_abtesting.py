import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def exploracion_datos(df, columna_control):

    print(f"\nEl número de datos es {df.shape[0]} y el de columnas es {df.shape[1]}")

    print(f"\nLos duplicados que tenemos en el conjunto de datos son: {df.duplicated().sum()}")
    
    print("\nLos nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(df.isnull().sum() / df.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])

    print(f"\nLos tipos de las columnas son:")
    display(pd.DataFrame(df.dtypes, columns = ["tipo_dato"]))

    print("\nLos valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = df.select_dtypes(include = "O")

    for col in dataframe_categoricas.columns:
        print(f"La columna {col} tiene las siguientes valore únicos:")
        display(pd.DataFrame(df[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in df[columna_control].unique():
        dataframe_filtrado = df[df[columna_control] == categoria]
    
        print("\n =============================================== \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print(f"\nLos principales estadísticos de las columnas numéricas para el {categoria} son: ")
        display(dataframe_filtrado.describe().T)
    
# --------------------ASUNCIONES -----------------------

def test_normalidad(df, columna_grupo, columna_metrica, alpha=0.05, num_iter=5):
    np.random.seed(42)

    datos = df.groupby(columna_grupo)[columna_metrica].agg(["mean", "std"])
  
    lista_grupos = datos.reset_index()[columna_grupo]
    lista_grupos = list(lista_grupos)

    # Comprobar el tamaño de las muestras de cada grupo y coger la menor
    lista_sizes = [df[df[columna_grupo]==grupo].shape[0] for grupo in lista_grupos]
    min_size = min(lista_sizes)


    if min_size > 30: metodo = "Kolmogorov"
    else: metodo = "Shapiro"
    print(f"Según el test de {metodo} se han obtenido los siguientes resultados:\n")
        
    # Para cada grupo vamos a coger una muestra de tamaño min_size varias veces y ver el p-valor
    for i in range(datos.shape[0]):
        valores_mayores_alpha =0
        valores_menores_alpha =0

        for j in range(num_iter):

            dist = np.random.normal(loc=datos.iloc[i,0], scale = datos.iloc[i,1], size = min_size)
            df_grupo = (df[df[columna_grupo]==lista_grupos[i]]).sample(min_size)

            if min_size > 30:
                _, pvalor = stats.kstest(df_grupo[columna_metrica], dist)
            else:
                _, pvalor = stats.shapiro(df_grupo[columna_metrica], dist)

            if pvalor>alpha: valores_mayores_alpha+=1
            else : valores_menores_alpha+=1
        
        print(f"Para el grupo {lista_grupos[i]} hemos obtenido:")
        print(f"    {valores_mayores_alpha} de los valores SÍ siguen una distribución normal.")
        print(f"    {valores_menores_alpha} de los valores NO siguen una distribución normal.")
        print("-----------------------------------------------------\n")

    
def hisplot_grupos(df, columna_grupo, columna_metrica):
    
    lista_grupos = list(df[columna_grupo].unique())
    if len(lista_grupos) % 2 == 0:
        filas = len(lista_grupos) // 2
        pares=True
    else:
        filas = len(lista_grupos) // 2 +1
        pares = False

    fig, axes = plt.subplots(filas,2, figsize=(15,10))
    axes= axes.flat

    for i,grupo in enumerate(lista_grupos):
        sns.histplot(df[df[columna_grupo]==grupo], x =columna_metrica, kde=True, ax=axes[i])
        axes[i].set_xlabel(grupo)
        axes[i].set_ylabel("Recuento")

    if not pares:
        fig.delaxes(axes[-1])

    plt.tight_layout()  # Ajustar el layout para que no se solapen
    plt.show()


def test_homoceidad(df, columna_grupo, columna_metrica, test="levene", alpha=0.05):
    lista_grupos = list(df[columna_grupo].unique())
    for grupo in lista_grupos:
        df_metrica = df[df[columna_grupo] == grupo][columna_metrica]
        globals()[grupo] = df_metrica
    
    if test.lower() == "levene":
        _, pvalor = stats.levene(*[globals()[var] for var in lista_grupos])

    elif test.lower() == "bartlett":
        _, pvalor = stats.bartlett(*[globals()[var] for var in lista_grupos])

    else:
        print("Por favor introduzca un tipo de test valido: levene o bartlett")
        return
    
    if pvalor > alpha:
        print(f"En la variable {columna_metrica} las varianzas son homogéneas entre grupos.")
    else:
        print(f"En la variable {columna_metrica} las varianzas NO son homogéneas entre grupos.")


# --------------------NO PARAMÉTRICAS -----------------------

def crear_df_grupos (df, col_grupo, col_metrica):

    for valor in df[col_grupo].unique():
        globals()[valor.lower()] = df[df[col_grupo]==valor][col_metrica]

    return list(df[col_grupo].unique())


def elegir_test(valores, dependencia=False):

    if len(valores) > 2:
        print("Ejecutando Kruskal...")
        _, pvalor = stats.kruskal(*[globals()[var.lower()] for var in valores])

    elif len(valores)==2 and dependencia==True:
        print("Ejecutando Wilcoxon...")
        _, pvalor = stats.wilcoxon(*[globals()[var.lower()] for var in valores])

    elif len(valores)==2 and dependencia==False:
        print("Ejecutando Mann-Whitney...")
        _, pvalor = stats.mannwhitneyu(*[globals()[var.lower()] for var in valores])

    else:
        print("Valores no válidos")
        return
    
    if pvalor < 0.05:
        print(f"El p-valor es {pvalor}, sí hay diferencías significativas entre los grupos.")
    else:
        print(f"El p-valor es {round(pvalor,4)}, NO hay diferencías significativas entre los grupos.")

