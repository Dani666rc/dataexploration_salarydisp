from turtle import width
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import tkinter as TK 
from turtle import width

# CARGAR LOS DATOS, CREAR Y LIMPIAR DATAFRAME BASE
def data():
    df_salario = pd.read_csv('data/Disparidad_Salarial_Hombres_Mujeres.csv')
    df_salario = pd.DataFrame(df_salario)


    # CREAR Y AJUSTAR NUEVO DATAFRAME 
    df_salario = df_salario[['Dominio geográfico', 'año', 'valor']]
    df_salario['Dominio geográfico'] = df_salario['Dominio geográfico'].replace('colombia', 'Colombia')



    df = df_salario.rename(columns={'Dominio geográfico':'Departamento', 'año':'Año', 'valor': 'Diferencia salario'})
    
    
    # Convertir la columna 'Monto' a valores numéricos (por ejemplo, enteros)
    
    df['Departamento'] = df['Departamento'].str.capitalize()
    df = df[df['Departamento'] != 'Colombia']

    # AGREGAR SMMLV POR AÑOS 
    smmlv_por_año = {2008: 461500, 2009: 496900, 2010: 515000, 2011: 535600, 2012: 566700, 2013: 589500, 2014: 616000, 2015: 644350,
                 2016: 689455, 2017: 737717, 2018: 781242, 2019: 828116, 2020: 877803}

    df['SMMLV'] = df['Año'].map(smmlv_por_año)

    # CALCULAR LA DIFERENCIA DE SALARIO COMO PORCENTAJE DEL SALARIO MINIMO PARA CADA AÑO
    df['Diferencia % SMMLV'] = ((df['Diferencia salario'] / df['SMMLV']) * 100).round(2)
    

    return df

# HISTORICO COLOMBIA
def historico_colombia():
    df_salario = pd.read_csv('data/Disparidad_Salarial_Hombres_Mujeres.csv')
    df_salario = pd.DataFrame(df_salario)


    # CREAR Y AJUSTAR NUEVO DATAFRAME 
    df_salario = df_salario[['Dominio geográfico', 'año', 'valor']]
    df_salario['Dominio geográfico'] = df_salario['Dominio geográfico'].replace('colombia', 'Colombia')

    df = df_salario.rename(columns={'Dominio geográfico':'Departamento', 'año':'Año', 'valor': 'Diferencia salario'})

    df['Departamento'] = df['Departamento'].str.capitalize()
    df = df[df['Departamento'] == 'Colombia']

    # AGREGAR SMMLV POR AÑOS 
    smmlv_por_año = {2008: 461500, 2009: 496900, 2010: 515000, 2011: 535600, 2012: 566700, 2013: 589500, 2014: 616000, 2015: 644350,
                 2016: 689455, 2017: 737717, 2018: 781242, 2019: 828116, 2020: 877803}

    df['SMMLV'] = df['Año'].map(smmlv_por_año)

    # CALCULAR LA DIFERENCIA DE SALARIO COMO PORCENTAJE DEL SALARIO MINIMO PARA CADA AÑO
    df['Diferencia % SMMLV'] = ((df['Diferencia salario'] / df['SMMLV']) * 100).round(2)

    df['Diferencia salario'] = df['Diferencia salario'].apply(lambda x: f"${x:,.0f}")
    df['SMMLV'] = df['SMMLV'].apply(lambda x: f"${x:,.0f}")
    
    # Crear un gráfico de líneas con Seaborn
    sns.set(style="dark")
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df, x='Año', y='Diferencia % SMMLV')
    plt.xlabel('Año')
    plt.ylabel('Valor como porcentaje del SMMLV')
    plt.title(f'Evolución de la disparidad salarial en Colombia')

    # Agregar etiquetas a los puntos en el gráfico
    for index, row in df.iterrows():
        plt.text(row['Año'], row['Diferencia % SMMLV'], str(round(row['Diferencia % SMMLV'], 2)), ha='right')
   
    
    # Mostrar el gráfico en la aplicación de Streamlit
    st.pyplot(plt)
    st.write(
        f"<div style='display: flex; justify-content: center;'><div>{df.to_html(index=False)}</div></div>",
        unsafe_allow_html=True
    )


# AGRUPAR POR DEPARTAMENTO

def agrupar_por_departamento():
    df = data()
    # Obtener la lista de departamentos únicos
    departamentos = df['Departamento'].unique()

    # Widget de selección para el usuario
    departamento_elegido = st.selectbox("Selecciona un departamento", departamentos)

    # Filtrar el DataFrame por el departamento elegido
    departamento_df = df[df['Departamento'] == departamento_elegido]
    departamento_df['Diferencia salario'] = df['Diferencia salario'].astype(str)

    # Formatear la columna 'Monto' como pesos colombianos
    departamento_df['Diferencia salario'] = df['Diferencia salario'].apply(lambda x: f"${x:,.0f}")
    departamento_df['SMMLV'] = departamento_df['SMMLV'].apply(lambda x: f"${x:,.0f}")
    # Crear un gráfico de líneas con Seaborn
    sns.set(style="dark")
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=departamento_df, x='Año', y='Diferencia % SMMLV')
    plt.xlabel('Año')
    plt.ylabel('Valor como porcentaje del SMMLV')
    plt.title(f'Evolución de la disparidad salarial para el departamento de {departamento_elegido}')

    # Agregar etiquetas a los puntos en el gráfico
    for index, row in departamento_df.iterrows():
        plt.text(row['Año'], row['Diferencia % SMMLV'], str(round(row['Diferencia % SMMLV'], 2)), ha='right')
   
    
    # Mostrar el gráfico en la aplicación de Streamlit
    st.pyplot(plt)
    st.write(
        f"<div style='display: flex; justify-content: center;'><div>{departamento_df.to_html(index=False)}</div></div>",
        unsafe_allow_html=True
    )

# AGRUPAR POR AÑO

def agrupar_por_año():
    df = data()
    # Obtener la lista de departamentos únicos
    años = df['Año'].unique()

    # Widget de selección para el usuario
    año_elegido = st.selectbox("Selecciona un año", años)

    # Filtrar el DataFrame por el departamento elegido
    año_df = df[df['Año'] == año_elegido]
    
    # Formatear la columna 'Monto' como pesos colombianos
    año_df['Diferencia salario'] = df['Diferencia salario'].apply(lambda x: f"${x:,.0f}")
    año_df['SMMLV'] = año_df['SMMLV'].apply(lambda x: f"${x:,.0f}")

    # Crear un gráfico de barras con Seaborn
    sns.set(style="dark")
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=año_df, x='Departamento', y='Diferencia % SMMLV')
    # Aumentar los márgenes del gráfico
    plt.xticks(fontsize=8, rotation=90)  # Etiquetas del eje x
    plt.xlabel('Departamento')
    plt.ylabel('Valor como porcentaje del SMMLV')
    plt.title(f'Gráfico de Barras para todos los departamentos para el año {año_elegido}')
    
    # Mostrar el gráfico de barras
    st.pyplot(plt)

    st.write(
        f"<div style='display: flex; justify-content: center;'><div>{año_df.to_html(index=False)}</div></div>",
        unsafe_allow_html=True
    )

# AGRUPAR MAXIMOS POR AÑO
def top_por_año():
    
    df = data()

    # Obtener la lista de departamentos únicos
    años = df['Año'].unique()

    # Widget de selección para el usuario
    año_elegido = st.selectbox("Selecciona un año", años, key="selectbox_year")

    # Filtrar el DataFrame por el departamento elegido
    año_df = df[df['Año'] == año_elegido]

    top10 = año_df.nlargest(10,'Diferencia % SMMLV')
    top10.index = pd.RangeIndex(start=1, stop=11, step=1)

    top10['Diferencia salario'] = df['Diferencia salario'].apply(lambda x: f"${x:,.0f}")
    top10['SMMLV'] = top10['SMMLV'].apply(lambda x: f"${x:,.0f}")

    # Crear un gráfico de barras con Seaborn
    sns.set(style="dark")
    sns.set_palette("colorblind")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=top10, x='Departamento', y='Diferencia % SMMLV', linewidth=0.5)
    # Aumentar los márgenes del gráfico
    plt.xticks(fontsize=10, rotation=45)  # Etiquetas del eje x
    plt.xlabel('Departamento')
    plt.ylabel('Valor como porcentaje del SMMLV')
    plt.title(f'Gráfico de Barras para los 10 departamentos con mayor disparidad salarial en el año {año_elegido}')
    
    # Mostrar el gráfico de barras
    st.pyplot(plt)

    
    st.write(
        f"<div style='display: flex; justify-content: center;'><div>{top10.to_html(index=True)}</div></div>",
        unsafe_allow_html=True
    )

def frecuencia_top10():
    
    data1 = data()
    df = pd.DataFrame(data1)

    # Crear un DataFrame para almacenar las diez primeras posiciones por año
    top10_por_año = pd.DataFrame()

    # Obtener la lista de años únicos
    años = df['Año'].unique()

    # Iterar a través de cada año
    for año in años:
        # Filtrar el DataFrame por el año actual y obtener las diez primeras posiciones
        top10 = df[df['Año'] == año].nlargest(10, 'Diferencia % SMMLV')
        top10_por_año = pd.concat([top10_por_año, top10]) 

    # Contar la frecuencia de aparición de cada departamento en el top 10 en todos los años
    frecuencia_total = top10_por_año['Departamento'].value_counts().reset_index()
    frecuencia_total.columns = ['Departamento', 'Frecuencia']   

    # Agregar departamentos que no aparecen en el top 10
    todos_departamentos = df['Departamento'].unique()
    departamentos_faltantes = [dep for dep in todos_departamentos if dep not in frecuencia_total['Departamento'].values]

    faltantes_df = pd.DataFrame({'Departamento': departamentos_faltantes, 'Frecuencia': 0})
    
    # Concatenar el DataFrame de frecuencia total con el de departamentos faltantes
    frecuencia_total = pd.concat([frecuencia_total, faltantes_df])
    frecuencia_total.index = range(1, len(frecuencia_total) + 1)
   
    st.write(
        f"<div style='display: flex; justify-content: center;'><div>{frecuencia_total.to_html(index=True)}</div></div>",
        unsafe_allow_html=True
    )


# ANÁLISIS EXPLORATORIO DE DATOS - INTERFAZ

st.title("Exploración de datos")
st.header("Disparidad salarial en Colombia y sus departamentos para los años 2008 - 2020")
st.write("La disparidad salarial de género sigue siendo una preocupación global en el ámbito laboral, con amplias implicaciones en la economía mundial. A pesar de los avances en la igualdad de género, las mujeres continúan enfrentando desafíos en sus ingresos, con remuneraciones inferiores en comparación con los hombres. Este fenómeno tiene un impacto directo en la estabilidad económica y la equidad en todo el mundo. \n \n La disparidad salarial no solo refleja desigualdades sistémicas, sino que también obstaculiza el crecimiento económico y la competitividad global. Abordar esta cuestión es esencial para promover la diversidad en el lugar de trabajo, impulsar la innovación y lograr un desarrollo sostenible en la economía global.")
image = "data/img.jpg"
st.image(image, caption="Disparidad salarial por género",use_column_width=True)
st.write("")
st.write("El gráfico que se presenta a continuación nos ofrece una visualización de los datos sobre la disparidad salarial entre hombres y mujeres en los años 2008 a 2020. Utiliza un porcentaje calculado que refleja la diferencia en pesos con respecto al salario mínimo legal vigente. Este enfoque nos permite comprender de manera efectiva cómo la brecha salarial ha evolucionado a lo largo del tiempo. \n\nObservar estas tendencias es esencial para abordar la igualdad de género en el ámbito laboral y respalda la toma de decisiones basadas en datos para promover un entorno más equitativo y justo en el mundo laboral.")

st.subheader("Colombia 2008 - 2020")
historico_colombia()

st.header("Disparidad salarial por Departamento")
st.write("Selecciona el departamento para el cual quieras conocer los datos")
agrupar_por_departamento()

st.header("Disparidad salarial por año")
st.write("Selecciona el año para el cual quieras conocer los datos (2008 hasta 2020)")
agrupar_por_año()

st.header("Top 10 departamentos por año")
st.write("Selecciona el año para el cual quieras conocer los diez departamentos que registraron una mayor disparidad salarial como porcentaje del salario mínimo legal vigente para dicho año.")
top_por_año()

st.subheader("Frecuencia top 10")
st.write("La siguiente tabla muestra la cantidad de veces que un departamento aparecio en el top 10 por año.")
frecuencia_top10()

st.write("")
st.write("")
st.write("")
st.write(f"Si quieres saber más acerca de la disparidad laboral entre géneros consulta esta infografia sobre la brecha de género en el empleo creada por la Organización Internacional del Trabajo. {'https://www.ilo.org/infostories/es-ES/Stories/Employment/barriers-women#intro'}")
st.write("")
st.write(f"Esta aplicación web para la exploración de datos se construyó utilizando los datos abiertos proporcionados por el Ministerio del Trabajo de Colombia y que se encuentran en el siguiente enlace: {'https://www.datos.gov.co/Trabajo/Disparidad-Salarial-Hombres-Mujeres/hf6d-emrx'} ")


