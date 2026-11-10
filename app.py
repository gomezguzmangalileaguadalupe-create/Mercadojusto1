"""Mercado Justo | Panel analítico para producción bovina.

Inicio: python app.py. Los datos reales van en data/base_carne_14000_datos.xlsx.
El archivo Excel incluido se carga como base de trabajo; los registros no son cotizaciones en vivo.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import folium
import gradio as gr
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get('MERCADO_DATA_PATH', str(BASE / 'data' / 'base_carne_14000_datos.xlsx')))
CAT = ['estado', 'producto', 'mercado', 'tipo_alimentacion']
NUM = ['inflacion', 'temperatura', 'anio', 'mes', 'dia_mes', 'dia_semana']
FEATURES = CAT + NUM
REQUIRED = CAT + ['fecha', 'precio_kg', 'inflacion', 'temperatura']
COORDS = {
    'CDMX': (19.4326, -99.1332, 50), 'Puebla': (19.0414, -98.2063, 130),
    'Jalisco': (20.6597, -103.3496, 550), 'Oaxaca': (17.0732, -96.7266, 460),
    'Sonora': (29.0892, -110.9612, 1600),
}
ROUTES = [('Sonora', 'Jalisco', 1300), ('Sonora', 'CDMX', 1600),
          ('Jalisco', 'CDMX', 540), ('Jalisco', 'Puebla', 670),
          ('CDMX', 'Puebla', 130), ('CDMX', 'Oaxaca', 460), ('Puebla', 'Oaxaca', 340)]
COLORS = {'ink': '#10253F', 'blue': '#2563eb', 'teal': '#0f766e', 'orange': '#c2410c'}


def number_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')


def load_data(path: Path | None = None) -> tuple[pd.DataFrame, bool, str]:
    """Lee datos con validación estricta y elimina registros incompletos."""
    source = path if path is not None else DATA_PATH
    if not source.is_file():
        raise FileNotFoundError(f'No existe {source}. Incluye data/base_carne_14000_datos.xlsx en el despliegue.')
    if source.suffix.lower() in ('.xlsx', '.xls'):
        raw = pd.read_excel(source)
    elif source.suffix.lower() == '.csv':
        raw = pd.read_csv(source)
    else:
        raise ValueError('Formato no compatible. Usa .xlsx o .csv.')
    raw.columns = raw.columns.astype(str).str.strip().str.lower()
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError('Faltan columnas en el archivo: ' + ', '.join(missing))
    df = raw.copy()
    for c in CAT:
        df[c] = df[c].astype('string').str.strip().str.lower()
    df['estado'] = df['estado'].replace({'ciudad de méxico': 'cdmx', 'ciudad de mexico': 'cdmx', 'méxico df': 'cdmx'})
    for c in ['inflacion', 'temperatura', 'precio_kg']:
        df[c] = number_series(df[c])
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce', dayfirst=False)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    df = df[df['precio_kg'] > 0].copy()
    if df.empty:
        raise ValueError('No quedaron filas válidas: revisa fechas, precios y columnas.')
    df['anio'] = df.fecha.dt.year
    df['mes'] = df.fecha.dt.month
    df['dia_mes'] = df.fecha.dt.day
    df['dia_semana'] = df.fecha.dt.dayofweek
    df = df.sort_values('fecha').reset_index(drop=True)
    return df, False, str(source)


DF, IS_DEMO, SOURCE = load_data()
PRODUCTS = sorted(DF.producto.unique().tolist())
STATES = sorted(DF.estado.unique().tolist())
MARKETS = sorted(DF.mercado.unique().tolist())
FEEDS = sorted(DF.tipo_alimentacion.unique().tolist())


def latest_price(product: str, state: str | None = None) -> float | None:
    part = DF[DF.producto.eq(product)]
    if state is not None:
        part = part[part.estado.eq(state)]
    return None if part.empty else float(part.iloc[-1].precio_kg)


def label_state(state: str) -> str:
    return 'CDMX' if state == 'cdmx' else state.title()


def card(title: str, value: str, note: str = '', accent: str = '#2563eb') -> str:
    return (f'<div class="metric" style="border-left:5px solid {accent}">'
            f'<div class="metric-label">{title}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-note">{note}</div></div>')


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
    fig.update_layout(height=370, template='plotly_white', xaxis_visible=False, yaxis_visible=False)
    return fig


def market_analysis(product: str, state: str):
    rows = DF[DF.producto.eq(product) & DF.estado.eq(state)].copy()
    if rows.empty:
        return card('Sin registros', '—', 'Prueba otra combinación.', '#c2410c'), empty_figure('Sin datos')
    # Agregar por fecha evita conectar puntos incongruentes de diferentes mercados en el mismo día.
    daily = rows.groupby('fecha', as_index=False)['precio_kg'].mean().sort_values('fecha')
    daily['media_7'] = daily.precio_kg.rolling(7, min_periods=1).mean()
    last = daily.iloc[-1]
    average = float(daily.precio_kg.mean())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.fecha, y=daily.precio_kg, name='Promedio diario',
                             mode='lines+markers', line=dict(color='#94a3b8', width=1), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=daily.fecha, y=daily.media_7, name='Media móvil (7 observaciones)',
                             mode='lines', line=dict(color=COLORS['blue'], width=3)))
    fig.add_hline(y=average, line_dash='dash', line_color=COLORS['teal'],
                  annotation_text=f'Promedio ${average:,.2f}')
    fig.update_layout(title=f'{product.title()} · {label_state(state)}', template='plotly_white',
                      height=390, hovermode='x unified', margin=dict(l=20, r=20, t=65, b=20),
                      xaxis_title='Fecha', yaxis_title='MXN / kg', legend=dict(orientation='h', y=-.3))
    return card('Último precio en los datos', f'${last.precio_kg:,.2f} MXN/kg',
                f'Fecha del registro: {last.fecha:%d/%m/%Y} · Media histórica ${average:,.2f}', COLORS['teal']), fig


@lru_cache(maxsize=1)
def get_model() -> Pipeline:
    """Entrenamiento diferido: la interfaz arranca sin esperar al bosque aleatorio."""
    prep = ColumnTransformer([
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), CAT),
        ('numericas', 'passthrough', NUM),
    ])
    model = Pipeline([('preprocesamiento', prep),
                      ('modelo', RandomForestRegressor(n_estimators=80, min_samples_leaf=2,
                                                      random_state=42, n_jobs=2))])
    model.fit(DF[FEATURES], DF.precio_kg)
    return model


def predict_price(product, state, market, feed, inflation, temperature, prediction_date):
    try:
        dt = pd.to_datetime(prediction_date, errors='raise')
        if pd.isna(dt):
            raise ValueError('Fecha vacía')
        input_df = pd.DataFrame([{
            'estado': state, 'producto': product, 'mercado': market,
            'tipo_alimentacion': feed, 'inflacion': float(inflation),
            'temperatura': float(temperature), 'anio': dt.year, 'mes': dt.month,
            'dia_mes': dt.day, 'dia_semana': dt.dayofweek,
        }])
        prediction = float(get_model().predict(input_df)[0])
        # No presenta predicciones como cotizaciones o precios reales.
        return card('Predicción del modelo', f'${prediction:,.2f} MXN/kg',
                    f'Proyección para {dt:%d/%m/%Y} · Random Forest · estimación, no cotización', COLORS['blue'])
    except (ValueError, TypeError) as exc:
        return card('Revisa la fecha', 'Fecha inválida', str(exc), '#c2410c')


def finance_analysis(product, fixed_cost, variable_cost, desired_profit):
    price = latest_price(product)
    if price is None:
        return card('Sin precio', '—', 'No hay registros del producto.'), empty_figure('Sin datos')
    fixed_cost, variable_cost, desired_profit = map(float, (fixed_cost, variable_cost, desired_profit))
    margin = price - variable_cost
    if margin <= 0:
        return card('Operación inviable', f'${margin:,.2f} MXN/kg',
                    'El precio registrado no cubre el costo variable por kilo.', '#b91c1c'), empty_figure('No existe punto de equilibrio')
    break_even = fixed_cost / margin
    target = (fixed_cost + desired_profit) / margin
    max_kg = max(100.0, target * 1.5)
    kg = np.linspace(0, max_kg, 120)
    revenues = kg * price
    expenses = fixed_cost + kg * variable_cost
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kg, y=revenues, mode='lines', name='Ingresos',
                             line=dict(color='#15803d', width=3)))
    fig.add_trace(go.Scatter(x=kg, y=expenses, mode='lines', name='Costos totales',
                             line=dict(color='#dc2626', width=3)))
    fig.add_trace(go.Scatter(x=kg, y=revenues, fill='tonexty', mode='lines',
                             line=dict(width=0), fillcolor='rgba(37,99,235,.08)',
                             name='Diferencia ingreso-costo', visible='legendonly'))
    fig.add_vline(x=break_even, line_dash='dash', line_color='#ea580c',
                  annotation_text=f'Equilibrio: {break_even:,.0f} kg')
    fig.add_vline(x=target, line_dash='dot', line_color='#2563eb',
                  annotation_text=f'Meta: {target:,.0f} kg')
    fig.update_layout(template='plotly_white', height=390,
                      title=f'Rentabilidad de {product.title()}', xaxis_title='Volumen (kg)',
                      yaxis_title='Pesos mexicanos (MXN)', margin=dict(l=20, r=20, t=65, b=20),
                      legend=dict(orientation='h', y=-.25))
    return (card('Precio de referencia', f'${price:,.2f} MXN/kg',
                 f'Equilibrio: {break_even:,.1f} kg · Meta: {target:,.1f} kg', COLORS['teal'])
            ), fig


def logistics_map(product, base_cost, diesel_cost):
    base_cost, diesel_cost = float(base_cost), float(diesel_cost)
    graph = nx.Graph()
    for a, b, distance in ROUTES:
        graph.add_edge(a, b, distance=distance, weight=distance * diesel_cost)
    mst = nx.minimum_spanning_tree(graph, weight='weight', algorithm='kruskal')
    best = None
    best_margin = -float('inf')
    for state, (lat, lon, hub_distance) in COORDS.items():
        price = latest_price(product, state.lower())
        if price is None:
            continue
        # Conserva la hipótesis original de 500 kg transportados.
        margin = price - base_cost - (hub_distance * diesel_cost / 500)
        if margin > best_margin:
            best_margin, best = margin, state
    mp = folium.Map(location=[23.6345, -102.5528], zoom_start=5, tiles='OpenStreetMap')
    for state, (lat, lon, hub_distance) in COORDS.items():
        price = latest_price(product, state.lower())
        price_text = 'Sin datos' if price is None else f'${price:,.2f} MXN/kg'
        folium.Marker([lat, lon], tooltip=state,
                      popup=folium.Popup(f'<b>{state}</b><br>Precio del conjunto: {price_text}<br>Distancia hub: {hub_distance} km', max_width=250),
                      icon=folium.Icon(color='green' if state == best else 'blue', icon='truck', prefix='fa')).add_to(mp)
    for a, b, values in mst.edges(data=True):
        folium.PolyLine([COORDS[a][:2], COORDS[b][:2]], color='#0f766e', weight=4,
                        tooltip=f'{a} — {b}: {values["distance"]} km (distancia estimada)').add_to(mp)
    total_cost = sum(v['weight'] for _, _, v in mst.edges(data=True))
    summary = card('Mejor margen estimado', best or 'Sin datos',
                   ('Margen neto estimado: ${:,.2f} MXN/kg · Costo de la red: ${:,.2f} MXN'.format(best_margin, total_cost)
                    if best else 'No hay precios para los estados del mapa.'), COLORS['teal'])
    return summary, mp._repr_html_()


CSS = """
.gradio-container {max-width: 1250px !important; margin: auto !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;}
.hero {padding: 27px 28px; border-radius: 18px; color: white; background: linear-gradient(115deg, #10253f, #155e75); margin: 8px 0 18px;}
.hero h1 {font-size: clamp(1.45rem, 3vw, 2.35rem); color: white !important; margin: 0 0 8px;}
.hero p {margin: 0; color: #dbeafe; font-size: 1rem;}
.metric {background: var(--block-background-fill); color: var(--body-text-color); border-radius: 12px; padding: 19px; margin: 10px 0 12px; box-shadow: 0 2px 8px rgba(15,23,42,.07);}
.metric-label {font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; opacity:.78; font-weight: 700;}
.metric-value {font-weight: 800; font-size: clamp(1.3rem, 3vw, 2.1rem); margin: 6px 0; overflow-wrap: anywhere;}
.metric-note {font-size: .86rem; opacity: .83;}
.data-warning {border-left: 4px solid #ca8a04; border-radius: 8px; padding: 12px 16px; background: color-mix(in srgb, #f59e0b 9%, transparent);}
footer {visibility: hidden;}
@media (max-width: 640px) {.hero {padding: 20px 16px;} .metric {padding: 14px;}}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title='Mercado Justo | Analítica bovina') as demo:
        gr.HTML('<section class="hero"><h1>🥩 Mercado Justo</h1>'
                '<p>Inteligencia de precios, predicción, rentabilidad y distribución bovina</p></section>')
        gr.Markdown(f'**Fuente de datos:** `{DATA_PATH.name}` · '
                    f'**Último registro:** {DF.fecha.max():%d/%m/%Y} · '
                    f'**Observaciones válidas:** {len(DF):,} · '
                    f'**Cobertura:** {len(STATES)} estados y {len(PRODUCTS)} productos. '
                    '**Los precios son históricos del archivo, no cotizaciones en tiempo real.**')
        with gr.Tab('📊 Mercado histórico'):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    p1 = gr.Dropdown(PRODUCTS, value=PRODUCTS[0], label='Corte bovino')
                    s1 = gr.Dropdown(STATES, value=STATES[0], label='Estado')
                    k1 = gr.HTML()
                with gr.Column(scale=2, min_width=320):
                    plot1 = gr.Plot(label='Histórico')
            for control in [p1, s1]:
                control.change(market_analysis, [p1, s1], [k1, plot1])
            demo.load(market_analysis, [p1, s1], [k1, plot1])
        with gr.Tab('🧠 Predicción IA'):
            with gr.Row():
                with gr.Column(min_width=265):
                    p2 = gr.Dropdown(PRODUCTS, value=PRODUCTS[0], label='Corte')
                    s2 = gr.Dropdown(STATES, value=STATES[0], label='Estado')
                    m2 = gr.Dropdown(MARKETS, value=MARKETS[0], label='Canal de venta')
                    f2 = gr.Dropdown(FEEDS, value=FEEDS[0], label='Alimentación')
                with gr.Column(min_width=265):
                    date2 = gr.Textbox(value=date.today().isoformat(), label='Fecha (AAAA-MM-DD)')
                    i2 = gr.Slider(0, 20, value=6, step=.1, label='Inflación (%)')
                    t2 = gr.Slider(-10, 50, value=24, step=.5, label='Temperatura (°C)')
                    predict = gr.Button('Calcular predicción', variant='primary')
                    k2 = gr.HTML()
            predict.click(predict_price, [p2, s2, m2, f2, i2, t2, date2], k2)
        with gr.Tab('📈 Finanzas'):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    p3 = gr.Dropdown(PRODUCTS, value=PRODUCTS[0], label='Corte')
                    fc = gr.Slider(0, 100000, value=15000, step=500, label='Costos fijos (MXN)')
                    vc = gr.Slider(0, 500, value=180, step=5, label='Costo variable (MXN/kg)')
                    gp = gr.Slider(0, 200000, value=30000, step=1000, label='Ganancia deseada (MXN)')
                    k3 = gr.HTML()
                with gr.Column(scale=2, min_width=320):
                    plot3 = gr.Plot()
            for control in [p3, fc, vc, gp]:
                control.change(finance_analysis, [p3, fc, vc, gp], [k3, plot3])
            demo.load(finance_analysis, [p3, fc, vc, gp], [k3, plot3])
        with gr.Tab('🗺️ Logística'):
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    p4 = gr.Dropdown(PRODUCTS, value=PRODUCTS[0], label='Producto')
                    bc = gr.Slider(0, 400, value=150, step=5, label='Costo base de producción (MXN/kg)')
                    dc = gr.Slider(0, 50, value=12, step=1, label='Costo de diésel (MXN/km)')
                    calculate_map = gr.Button('Actualizar mapa', variant='primary')
                    k4 = gr.HTML()
                with gr.Column(scale=2, min_width=320):
                    mp = gr.HTML()
            calculate_map.click(logistics_map, [p4, bc, dc], [k4, mp])
            demo.load(logistics_map, [p4, bc, dc], [k4, mp])
    return demo


if __name__ == '__main__':
    build_app().launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', '10000')),
                       show_error=True, theme=gr.themes.Soft(), css=CSS)
