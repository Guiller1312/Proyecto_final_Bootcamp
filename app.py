import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def main():
    
    # st.title("Análisis de Diagnóstico Médico")

    # # Paso 1: Cargar datos y preprocesar
    # st.header("Cargar y preprocesar datos")

    # # Cargar processed_data_with_diagnosis.csv
    # uploaded_file = st.file_uploader("Subir archivo CSV con los datos", type="csv")

    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file)

    #     # Limpieza básica y manejo de datos faltantes
    #     data.dropna(inplace=True)

    #     st.write("Datos cargados y preprocesados:")
    #     st.write(data.head())

    #     # Paso 2: Seleccionar características y objetivo
    #     features = ['PatientGender', 'Age', 'PatientRace', 'CBC: ABSOLUTE LYMPHOCYTES', 'CBC: ABSOLUTE NEUTROPHILS',
    #                 'CBC: BASOPHILS', 'CBC: EOSINOPHILS', 'CBC: HEMATOCRIT', 'CBC: HEMOGLOBIN', 'CBC: LYMPHOCYTES',
    #                 'CBC: MCH', 'CBC: MCHC', 'CBC: MEAN CORPUSCULAR VOLUME', 'CBC: MONOCYTES', 'CBC: NEUTROPHILS',
    #                 'CBC: PLATELET COUNT', 'CBC: RDW', 'CBC: RED BLOOD CELL COUNT', 'CBC: WHITE BLOOD CELL COUNT',
    #                 'METABOLIC: ALBUMIN', 'METABOLIC: ALK PHOS', 'METABOLIC: ALT/SGPT', 'METABOLIC: ANION GAP',
    #                 'METABOLIC: AST/SGOT', 'METABOLIC: BILI TOTAL', 'METABOLIC: BUN', 'METABOLIC: CALCIUM',
    #                 'METABOLIC: CARBON DIOXIDE', 'METABOLIC: CHLORIDE', 'METABOLIC: CREATININE', 'METABOLIC: GLUCOSE',
    #                 'METABOLIC: POTASSIUM', 'METABOLIC: SODIUM', 'METABOLIC: TOTAL PROTEIN', 'URINALYSIS: PH',
    #                 'URINALYSIS: RED BLOOD CELLS', 'URINALYSIS: SPECIFIC GRAVITY', 'URINALYSIS: WHITE BLOOD CELLS']

    #     target_code = 'PrimaryDiagnosisCode_y'
    #     target_description = 'PrimaryDiagnosisDescription_y'

    #     # Paso 3: Dividir en conjunto de entrenamiento y prueba
    #     X = data[features]
    #     y_code = data[target_code]
    #     y_description = data[target_description]

    #     X_train, X_test, y_train_code, y_test_code = train_test_split(X, y_code, test_size=0.2, random_state=42)
    #     _, _, y_train_description, y_test_description = train_test_split(X, y_description, test_size=0.2, random_state=42)

    #     # Paso 4: Codificación One-Hot de características categóricas
    #     X_train_encoded = pd.get_dummies(X_train)
    #     X_test_encoded = pd.get_dummies(X_test)

    #     # Asegurar que X_train y X_test tengan las mismas columnas después de la codificación
    #     missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
    #     for col in missing_cols:
    #         X_test_encoded[col] = 0
    #     X_test_encoded = X_test_encoded[X_train_encoded.columns]

    #     # Paso 5: Entrenar un modelo (Random Forest Classifier)
    #     st.header("Entrenar el modelo")
    #     model_code = RandomForestClassifier(random_state=42)
    #     model_code.fit(X_train_encoded, y_train_code)

    #     model_description = RandomForestClassifier(random_state=42)
    #     model_description.fit(X_train_encoded, y_train_description)

    #     # Paso 6: Evaluar el modelo
    #     st.header("Evaluar el modelo")
    #     # TO-DO

    #     # Paso 7: Predicción de nuevos datos (ejemplo)
    #     st.header("Predicción de nuevos datos")
    #     X_new = pd.DataFrame({
    #         'PatientGender': ['Male'],
    #         'Age': [103],
    #         'PatientRace': ['Asian'],
    #         'CBC: ABSOLUTE LYMPHOCYTES': [29.6],
    #         'CBC: ABSOLUTE NEUTROPHILS': [67.5],
    #         'CBC: BASOPHILS': [0.1],
    #         'CBC: EOSINOPHILS': [0.2],
    #         'CBC: HEMATOCRIT': [40.7],
    #         'CBC: HEMOGLOBIN': [15.9],
    #         'CBC: LYMPHOCYTES': [4.7],
    #         'CBC: MCH': [38.9],
    #         'CBC: MCHC': [30.2],
    #         'CBC: MEAN CORPUSCULAR VOLUME': [97.5],
    #         'CBC: MONOCYTES': [0.2],
    #         'CBC: NEUTROPHILS': [8.4],
    #         'CBC: PLATELET COUNT': [305.3],
    #         'CBC: RDW': [10.4],
    #         'CBC: RED BLOOD CELL COUNT': [5.6],
    #         'CBC: WHITE BLOOD CELL COUNT': [11.6],
    #         'METABOLIC: ALBUMIN': [3.3],
    #         'METABOLIC: ALK PHOS': [44.8],
    #         'METABOLIC: ALT/SGPT': [39.9],
    #         'METABOLIC: ANION GAP': [8.4],
    #         'METABOLIC: AST/SGOT': [14.7],
    #         'METABOLIC: BILI TOTAL': [0.3],
    #         'METABOLIC: BUN': [17.1],
    #         'METABOLIC: CALCIUM': [8.7],
    #         'METABOLIC: CARBON DIOXIDE': [25.5],
    #         'METABOLIC: CHLORIDE': [109.1],
    #         'METABOLIC: CREATININE': [0.6],
    #         'METABOLIC: GLUCOSE': [110.5],
    #         'METABOLIC: POTASSIUM': [5.3],
    #         'METABOLIC: SODIUM': [146.6],
    #         'METABOLIC: TOTAL PROTEIN': [6.4],
    #         'URINALYSIS: PH': [5.5],
    #         'URINALYSIS: RED BLOOD CELLS': [2.2],
    #         'URINALYSIS: SPECIFIC GRAVITY': [1.0],
    #         'URINALYSIS: WHITE BLOOD CELLS': [0.2]
    #     })

    #     st.subheader("Datos de entrada para la predicción")
    #     st.write(X_new)

    #     # Codificar las características categóricas de X_new
    #     X_new_encoded = pd.get_dummies(X_new)

    #     # Asegurar que X_new_encoded tenga las mismas columnas que X_train_encoded
    #     missing_cols_new = set(X_train_encoded.columns) - set(X_new_encoded.columns)
    #     for col in missing_cols_new:
    #         X_new_encoded[col] = 0
    #     X_new_encoded = X_new_encoded[X_train_encoded.columns]

    #     # Realizar predicciones
    #     predicted_code = model_code.predict(X_new_encoded)
    #     predicted_description = model_description.predict(X_new_encoded)

    #     st.subheader("Predicción para los nuevos datos:")
    #     st.write("Predicted PrimaryDiagnosisCode_y:", predicted_code[0])
    #     st.write("Predicted PrimaryDiagnosisDescription_y:", predicted_description[0])
    
    
    
    # Título de la aplicación
    st.title("Streamlit de 100 pacientes")

    # Carga el archivo PKL usando Streamlit
    with open("model_100_tests.pkl", "br") as file:
        model = pickle.load(file)
    with open("x_scaler.pkl", "br") as file:
        scaler = pickle.load(file)


    
    df = pd.read_csv("merged_df.csv")

    # SelectBox
    columnas = ['Age', 'CBC: ABSOLUTE LYMPHOCYTES (%)', 'CBC: ABSOLUTE NEUTROPHILS (%)', 
                        'CBC: BASOPHILS (k/cumm)', 'CBC: EOSINOPHILS (k/cumm)', 'CBC: HEMATOCRIT (%)', 'CBC: HEMOGLOBIN (gm/dl)', 
                        'CBC: LYMPHOCYTES (k/cumm)', 'CBC: MCH (pg)', 'CBC: MCHC (g/dl)', 'CBC: MEAN CORPUSCULAR VOLUME (fl)', 
                        'CBC: MONOCYTES (k/cumm)', 'CBC: NEUTROPHILS (k/cumm)', 'CBC: PLATELET COUNT (k/cumm)', 'CBC: RDW (%)', 
                        'CBC: RED BLOOD CELL COUNT (m/cumm)', 'CBC: WHITE BLOOD CELL COUNT (k/cumm)', 'METABOLIC: ALBUMIN (gm/dL)',
                        'METABOLIC: ALK PHOS (U/L)', 'METABOLIC: ALT/SGPT (U/L)', 'METABOLIC: ANION GAP (mmol/L)',
                        'METABOLIC: AST/SGOT (U/L)', 'METABOLIC: BILI TOTAL (mg/dL)', 'METABOLIC: BUN (mg/dL)', 
                        'METABOLIC: CALCIUM (mg/dL)', 'METABOLIC: CARBON DIOXIDE (mmol/L)', 
                        'METABOLIC: CHLORIDE (mmol/L)', 'METABOLIC: CREATININE (mg/dL)',
                        'METABOLIC: GLUCOSE (mg/dL)', 'METABOLIC: POTASSIUM (mmol/L)', 'METABOLIC: SODIUM (mmol/L)', 
                        'METABOLIC: TOTAL PROTEIN (gm/dL)', 'URINALYSIS: PH (no unit)',
                        'URINALYSIS: RED BLOOD CELLS (rbc/hpf)', 'URINALYSIS: SPECIFIC GRAVITY (no unit)', 
                        'URINALYSIS: WHITE BLOOD CELLS (wbc/hpf)']
    
    
    
    
    choice = st.multiselect(label = "Modulo", options = columnas, max_selections = 2, default = columnas[:2])
    choice_color = st.multiselect(label = "Modulo", options = columnas, max_selections = 1, default = columnas[:1])
    st.write(f"Columnas: {choice}")
    
    
    st.write(f"{choice[0]} vs {choice[1]}")
    fig, ax = plt.subplots()
    scatter = ax.scatter(x=df[choice[0]], y=df[choice[1]], alpha=0.7)
    plt.title(f" {choice[0]} vs {choice[1]}")
    plt.xlabel(choice[0])
    plt.ylabel(choice[1])
    # Añadir una leyenda
    legend1 = ax.legend(*scatter.legend_elements(), title="Prueba")
    ax.add_artist(legend1)
    st.pyplot(fig)
    
    # Gráfico de conteo con Seaborn
    st.write("Prueba")
    fig = plt.figure()
    sns.countplot(x=df["'CBC: HEMATOCRIT (%)'"], palette="viridis")
    plt.title("Distribución de 'CBC: HEMATOCRIT (%)'")
    st.pyplot(fig)
    
    

    fig_bar = px.bar(data_frame = df,
                    x          = choice[0],
                    y          = choice[1],
                    color      = choice_color[0]
                    )
    st.plotly_chart(figure_or_data = fig_bar)


if __name__ == "__main__":
    main()
