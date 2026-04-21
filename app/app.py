import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Dashboard de Predicción", layout="wide")

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('models/modelo_final.pkl')
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
        return model, df
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        return None, None

model, df_raw = load_assets()

# --- LÓGICA DE SIMULACIÓN ---
def run_simulation(df_prod, model, discount_adj, comp_scenario):
    df_sim = df_prod.sort_values('fecha').copy().reset_index(drop=True)
    features = list(model.feature_names_in_)
    
    # --- 1. SEGURIDAD PARA LAGS (Convertidos a FLOAT) ---
    for i in range(1, 8):
        col_name = f'unidades_vendidas_lag_{i}'
        # Forzamos que la columna sea float64 para aceptar decimales
        df_sim[col_name] = df_sim[col_name].astype(float) if col_name in df_sim.columns else 0.0
        
    if 'unidades_vendidas_ma7' not in df_sim.columns:
        df_sim['unidades_vendidas_ma7'] = 0.0
    else:
        df_sim['unidades_vendidas_ma7'] = df_sim['unidades_vendidas_ma7'].astype(float)

    # --- 2. MANEJO DE COMPETENCIA (_x) ---
    comp_cols = ['Amazon_x', 'Decathlon_x', 'Deporvillage_x']
    existing_comp = [c for c in df_sim.columns if c in comp_cols]
    
    if existing_comp:
        for col in existing_comp:
            df_sim[col] = df_sim[col].astype(float) * (1 + comp_scenario/100)
        df_sim['precio_competencia'] = df_sim[existing_comp].mean(axis=1)
    
    if 'precio_venta' in df_sim.columns:
        df_sim['precio_venta'] = df_sim['precio_venta'].astype(float) * (1 - discount_adj/100)

    # --- 3. BUCLE DE PREDICCIÓN ---
    df_sim['prediccion_final'] = 0.0
    predictions = []

    for i in range(len(df_sim)):
        if i > 0:
            for j in range(7, 1, -1):
                df_sim.loc[i, f'unidades_vendidas_lag_{j}'] = float(df_sim.loc[i-1, f'unidades_vendidas_lag_{j-1}'])
            df_sim.loc[i, 'unidades_vendidas_lag_1'] = float(predictions[i-1])
            
            hist = predictions[max(0, i-7):i]
            if len(hist) < 7:
                needed = 7 - len(hist)
                pasts = [df_sim.loc[i, f'unidades_vendidas_lag_{k}'] for k in range(1, needed + 1)]
                hist = pasts + hist
            df_sim.loc[i, 'unidades_vendidas_ma7'] = np.mean(hist)

        X_input = df_sim.loc[[i], features]
        pred = max(0, model.predict(X_input)[0])
        predictions.append(pred)
        df_sim.at[i, 'prediccion_final'] = float(pred)

    df_sim['ingresos_proyectados'] = df_sim['prediccion_final'] * df_sim['precio_venta']
    return df_sim

# --- INTERFAZ ---
if df_raw is not None:
    with st.sidebar:
        st.header("🎮 Controles de Simulación")
        # Buscamos todas las columnas que empiecen por nombre_
        all_cols = df_raw.columns.tolist()
        prod_cols = [c for c in all_cols if c.startswith('nombre_')]
        nombres = sorted([c.replace('nombre_', '') for c in prod_cols])
        
        sel_prod = st.selectbox("📦 Selecciona el producto", nombres)
        adj_desc = st.slider("💰 Ajuste de Descuento", -50, 50, 0)
        
        st.write("🏭 Escenario de Competencia")
        escenario = st.radio("", ["Actual (0%)", "-5%", "+5%"], label_visibility="collapsed")
        esc_val = {"Actual (0%)": 0, "-5%": -5, "+5%": 5}[escenario]
        
        btn_run = st.button("🚀 Simular Ventas", use_container_width=True)

    st.title("📊 Dashboard de Predicción - Noviembre 2025")
    st.subheader(f"Producto: {sel_prod}")

    # --- FILTRADO ULTRA-ROBUSTO ---
    target_col = f'nombre_{sel_prod}'
    
    # Intentamos filtrar por la columna nombre_...
    if target_col in df_raw.columns:
        df_target = df_raw[df_raw[target_col] >= 0.5].copy() # Usamos 0.5 por si es float 1.0
    else:
        df_target = pd.DataFrame()

    if not df_target.empty:
        # Tarjetas de Información rápida
        c1, c2, c3 = st.columns(3)
        c1.metric("Categoría", "Deportes")
        c2.metric("Subcategoría", "General")
        c3.metric("Precio Base", f"€{df_target['precio_venta'].iloc[0]:.2f}")

        if btn_run:
            with st.spinner('Simulando escenarios...'):
                res_df = run_simulation(df_target, model, adj_desc, esc_val)
            
            # Gráfico
            st.subheader("📈 Proyección de Ventas")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=res_df, x=res_df['fecha'].dt.day, y='prediccion_final', marker='o', color='#764ba2')
            plt.axvline(28, color='red', linestyle='--', label='Black Friday')
            st.pyplot(fig)

            # Resultados Numéricos
            m1, m2 = st.columns(2)
            m1.success(f"**Unidades Totales:** {int(res_df['prediccion_final'].sum())}")
            m2.info(f"**Ingresos Est.:** €{res_df['ingresos_proyectados'].sum():,.2f}")
    else:
        st.error(f"Error: No se encontraron datos activos para '{sel_prod}'.")
        st.write("Asegúrate de que la columna", target_col, "tenga valores '1' en tu CSV.")