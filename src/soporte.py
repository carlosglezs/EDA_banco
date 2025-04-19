import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)

def eda_preliminar (df):
    """
    Realiza un análisis exploratorio preliminar del DataFrame."
    df: "DataFrame de pandas a analizar.
    """
    display(df.sample(5))

    print('-----------------')
    
    print('INFO')

    display(df.info())

    print('-----------------')

    print('NULOS')

    display(round(df.isnull().sum()/df.shape[0]*100,2))

    print('-----------------')

    print('DUPLICADOS')

    print(df.duplicated().sum())

    print('-----------------')

    print('VALUE COUNTS')

    for col in df.select_dtypes(include='O').columns:
        print(df[col].value_counts())
        print('----------------------------')

def valores_minus(df):
    """
    Convierte todos los valores en columnas categóricas a minúsculas."
    df: "DataFrame con columnas tipo objeto.
    """
    for col in df.select_dtypes(include='O').columns:
        df[col] = df[col].str.lower()

def comas (df,lista_col):
    """
    Reemplaza comas por puntos en las columnas especificadas.
    df: DataFrame de entrada.
    lista_col: Lista de nombres de columnas a modificar.
    """
    for col in lista_col:
        df [col] = df[col].str.replace(',','.')

def convertir_fecha(fecha):
    """
    Convierte una fecha con nombre de mes en español a formato datetime.
    fecha: Cadena con la fecha en formato día-mes-año.
    """
    
    meses_es_en = {
        "enero": "january", "febrero": "february", "marzo": "march", "abril": "april",
        "mayo": "may", "junio": "june", "julio": "july", "agosto": "august",
        "septiembre": "september", "octubre": "october", "noviembre": "november", "diciembre": "diciembre"
    }
    
    if pd.isna(fecha) or not isinstance(fecha, str):  # Manejo de valores nulos o no strings
        return pd.NaT  # Devolver NaT en lugar de None para mantener datetime

    for mes_es, mes_en in meses_es_en.items():
        fecha = fecha.replace(mes_es, mes_en)  # Reemplazar mes en español por inglés

    return pd.to_datetime(fecha, format='%d-%B-%Y', errors='coerce')  # No usar .date()

def convertir_a_booleanos(df, columnas_bool):
    """
    Convierte columnas binarias (0/1) a valores de texto ('no'/'yes')."
    df: "DataFrame con las columnas booleanas."
    columnas_bool: "Lista de columnas a convertir.
    """
    for col in columnas_bool:
        try:
            df[col] = df[col].replace({0:'no',1:'yes'})
        except Exception:
            pass  # Si ocurre algún error, se deja la columna sin modificar

def cambiar_tipos(df, columnas_cambiar):
    """
    Convierte las columnas especificadas a tipo float64."
    df: "DataFrame con las columnas a modificar."
    columnas_cambiar: "Lista de nombres de columnas.
    """
    for col in columnas_cambiar:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass  # Si hay algún error en la conversión, se omite y se mantiene el valor original        

def convert_column_name(col_name):
    """
    Convierte nombres de columnas de CamelCase o PascalCase a snake_case,
    evitando dobles guiones bajos.
    col_name: "Nombre de columna en formato CamelCase o PascalCase."
    """
    col_name = re.sub(r'(?<!^)(?<!_)(?=[A-Z])', '_', col_name)  # Inserta "_" antes de mayúsculas (si no hay "_")
    return col_name.lower()  # Convertir todo a minúsculas


def calcular_nulos(dataframe):
    """
    Calcula el número y porcentaje de valores nulos por columna."
    dataframe: DataFrame a evaluar.
    """
    numero_nulos = dataframe.isnull().sum() 
    porcentaje_nulos = (dataframe.isnull().sum() / dataframe.shape[0]) * 100
    return numero_nulos, porcentaje_nulos

def analisis_general_cat(dataframe):
    """
    Analiza columnas categóricas: distribución, frecuencia y resumen."
    dataframe: DataFrame a evaluar.
    """
    col_cat = dataframe.select_dtypes(include="O").columns

    if len(col_cat) == 0:
        print("No hay columnas categoricas")
    else:
        for col in col_cat:
            print(f"La distribución de la columna {col.upper()}")
            print(f"Esta columna tiene longitud de {len(dataframe[col].unique())} valores únicos")
            display(dataframe[col].value_counts(normalize=True))
            print("-----------------------------\n Describe")
            display(dataframe[col].describe())
            print("-----------------------------")
            
    return col_cat


def calcular_solo_col_nul(dataframe, umbral = 10):
    """
    Identifica columnas con valores nulos y las separa por umbral.
    dataframe: DataFrame a evaluar.
    umbral: Porcentaje límite para diferenciar columnas con pocos o muchos nulos.
    """
    columns_with_nulls = dataframe.columns[dataframe.isnull().any()]
    nulls_columns_info = pd.DataFrame({
        "Column":columns_with_nulls,
        "Datatype":[dataframe[col].dtype for col in columns_with_nulls],
        "NullCount":[dataframe[col].isnull().sum() for col in columns_with_nulls],
        "Null%":[((dataframe[col].isnull().sum() / dataframe.shape[0]) * 100)for col in columns_with_nulls]
    })

    display (nulls_columns_info)
    high_nulls_cols = nulls_columns_info[nulls_columns_info['Null%']> umbral] ['Column'].tolist()
    low_nulls_cols = nulls_columns_info[nulls_columns_info['Null%']<= umbral] ['Column'].tolist()
    return high_nulls_cols, low_nulls_cols
    
def subplot_col_cat(dataframe):
    """
    Genera gráficos de barras para columnas categóricas.
    dataframe: DataFrame con columnas categóricas.
    """

    #Seleccionar columnas categoricas
    categorical_cols = dataframe.select_dtypes(include=['object','category']).columns

    if len(categorical_cols) == 0:
        print("No hay columnas categoricas en el dataframe")
        return

    #Configurar el tamaño de la figura
    num_cols = len(categorical_cols)
    rows = (num_cols +2) //3 #Calcular filas necesarias para 3 columnas por fila
    fig, axes = plt.subplots(rows, 3, figsize= (15,rows *5))
    axes = axes.flatten()  
    
    #Si solo le pasamos una variable
    if num_cols == 1:
        axes = [axes]  

    #Genenrar graficos para cada columna categorica
    for i, col in enumerate(categorical_cols):
        sns.countplot(data=dataframe, x=col, ax=axes[i], legend=False)
        axes[i].set_title(f"distribucion de {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x',rotation = 90) #Rotación de ser necesario

    #eliminar ejes sobrantes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])    

def subplot_col_num(dataframe,col):
    """
    Genera histogramas y boxplots para columnas numéricas.
    dataframe: DataFrame de entrada.
    col: Lista de columnas numéricas a graficar.
    """

    num_graph = len(col)
    num_rows = (num_graph + 2) // 2

    fig, axes = plt.subplots(num_graph, 2, figsize=(15, num_rows*5 ))

    for i, col in enumerate(col):
        sns.histplot(data=dataframe, x=col, ax = axes[i,0], bins = 200)
        axes[i,0].set_title(f"Distriucion de {col}")
        axes[i,0].set_ylabel("frecuencia")

        sns.boxplot(data=dataframe, x=col, ax = axes[i,1])
        axes[i,1].set_title(f"Boxplot de {col}")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def outliers_derecha(data,columnas):
    """
    Muestra la cantidad y porcentaje de outliers por columna (valores superiores a un umbral).
    data: DataFrame de entrada.
    columnas: Diccionario con nombre de columna como clave y umbral como valor.
    """
    for col, out in columnas.items():
        outliers = data[col] [data[col]>out].count()
        print(f"para la columna {col.upper()} tenemos {outliers}, lo que representa un {round(outliers/data.shape[0]*100,4)}%")   

def imputar_knn(data, lista_columnas):
    """
    Imputa valores nulos con el método KNN y agrega columnas nuevas con sufijo '_knn'.
    data: DataFrame con valores faltantes.
    lista_columnas: "Columnas a imputar.
    """

    knn_imputer = KNNImputer(n_neighbors=5)
    data_imputed = knn_imputer.fit_transform(data[lista_columnas])
    new_col = [col + "_knn" for col in lista_columnas]

    data[new_col] = data_imputed
    display(data[new_col].describe().T)
    return data, new_col

def imputar_iterative(data, lista_columnas):
    """
    Imputa valores nulos con imputación iterativa y agrega columnas con sufijo '_iterative'."
    data: DataFrame con valores faltantes.
    lista_columnas: Columnas a imputar.
    """
    iter_imputer = IterativeImputer(max_iter=50, random_state=42)
    data_imputed = iter_imputer.fit_transform(data[lista_columnas])
    new_col = [col + "_iterative" for col in lista_columnas]

    data[new_col] = data_imputed
    display(data[new_col].describe().T)
    return data, new_col        

def age_group(age):
    """
    Clasifica la edad en tres grupos: 'Joven (<30)', 'Adulto (30-60)' y 'Mayor (>60)'."
    age: Valor numérico de edad.
    """
    if age < 30:
        return 'Joven (<30)'
    elif age <= 60:
        return 'Adulto (30-60)'
    else:
        return 'Mayor (>60)'