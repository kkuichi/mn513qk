import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    PALETTE,
    VYSLEDOK_COL,
    apply_chart_theme,
    load_data,
    render_sidebar_filters,
    setup_page,
)

setup_page("Interaktívne grafy")

st.title("Interaktívne grafy")

df = load_data()
df_filtered, vek_options = render_sidebar_filters(df)

# -----------------------------------------------
# CARDS - POČTY PACIENTOV
# -----------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] p { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vybraní pacienti", len(df_filtered))
with col2:
    st.metric("Celkovo pacientov", len(df))
with col3:
    if len(df) > 0:
        st.metric("Podiel vybraných pacientov", f"{round(len(df_filtered)/len(df)*100, 1)} %")

st.markdown("---")

# -----------------------------------------------
# POHLAVIE + VÝSLEDOK HOSPITALIZÁCIE
# -----------------------------------------------
col4, col5 = st.columns([1, 2])

with col4:
    st.subheader("Pohlavie")
    pohlavie_counts = df_filtered['Pohlavie'].value_counts().reset_index()
    pohlavie_counts.columns = ['Pohlavie', 'Počet']
    total_pohlavie = pohlavie_counts['Počet'].sum()
    if total_pohlavie > 0:
        pohlavie_counts['Podiel'] = (pohlavie_counts['Počet'] / total_pohlavie) * 100
    else:
        pohlavie_counts['Podiel'] = 0

    fig_donut = px.pie(
        pohlavie_counts,
        names='Pohlavie',
        values='Počet',
        hole=0.45,
        color_discrete_sequence=[PALETTE['accent'], PALETTE['accent_light'], '#2D728F', '#84A59D']
    )
    fig_donut.update_traces(
        sort=False,
        customdata=pohlavie_counts[['Podiel']].to_numpy(),
        texttemplate='%{label}<br>%{value} (%{customdata[0]:.1f} %)',
        hovertemplate='%{label}: %{value} (%{customdata[0]:.1f} %)<extra></extra>',
        textposition='inside',
        insidetextorientation='horizontal'
    )
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.18,
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(l=18, r=18, t=8, b=78)
    )
    apply_chart_theme(fig_donut)
    st.plotly_chart(fig_donut, use_container_width=True)

with col5:
    st.subheader("Výsledok hospitalizácie")
    zavaznost_map = {1: '1 - Prepustenie domov', 2: '2 - Presun na oddelenie', 3: '3 - Smrť'}
    df_vysledok = df_filtered.copy()
    df_vysledok.loc[:, 'Výsledok_hospitalizácie_popis'] = df_vysledok[VYSLEDOK_COL].map(zavaznost_map)
    pocty_zavaznost = df_vysledok['Výsledok_hospitalizácie_popis'].value_counts().reset_index()
    pocty_zavaznost.columns = ['Výsledok hospitalizácie', 'Počet']
    total_zavaznost = pocty_zavaznost['Počet'].sum()
    if total_zavaznost > 0:
        pocty_zavaznost['Podiel'] = (pocty_zavaznost['Počet'] / total_zavaznost) * 100
        pocty_zavaznost['Text'] = (
            pocty_zavaznost['Počet'].astype(int).astype(str)
            + " ("
            + pocty_zavaznost['Podiel'].round(1).astype(str)
            + " %)"
        )
    else:
        pocty_zavaznost['Podiel'] = 0
        pocty_zavaznost['Text'] = pd.Series(dtype=str)
    fig_bar = px.bar(
        pocty_zavaznost,
        x='Počet',
        y='Výsledok hospitalizácie',
        orientation='h',
        color='Výsledok hospitalizácie',
        text='Text',
        color_discrete_map={
            '1 - Prepustenie domov': '#2ecc71',
            '2 - Presun na oddelenie': '#f39c12',
            '3 - Smrť': '#e74c3c'
        }
    )
    fig_bar.update_traces(textposition='outside', cliponaxis=False)
    max_text_len = int(pocty_zavaznost['Text'].str.len().max()) if not pocty_zavaznost.empty else 0
    adaptive_right_margin = max(50, min(260, 14 + (max_text_len * 6)))
    fig_bar.update_layout(
        showlegend=True,
        legend=dict(
            orientation='h',
            x=0.5,
            y=-0.24,
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(l=18, r=adaptive_right_margin, t=8, b=78),
        autosize=True
    )
    apply_chart_theme(fig_bar)
    fig_bar.update_xaxes(
        showgrid=True,
        zeroline=False,
        automargin=True,
        tick0=0,
        dtick=500,
        tickfont=dict(color=PALETTE['text']),
        title_font=dict(color=PALETTE['text'])
    )
    fig_bar.update_yaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
        tickfont=dict(color=PALETTE['text']),
        title_font=dict(color=PALETTE['text']),
        title_text=""
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------
# DISTRIBÚCIA VEKOVÝCH KATEGÓRIÍ
# -----------------------------------------------
st.subheader("Vek")
poradie = vek_options
df_vek_distribucia = df_filtered['Vek_kat'].value_counts().reset_index()
df_vek_distribucia.columns = ['Vek_kat', 'Počet']
total_vek_distribucia = df_vek_distribucia['Počet'].sum()
if total_vek_distribucia > 0:
    df_vek_distribucia['Podiel'] = (df_vek_distribucia['Počet'] / total_vek_distribucia) * 100
    df_vek_distribucia['Text'] = (
        df_vek_distribucia['Počet'].astype(int).astype(str)
        + " ("
        + df_vek_distribucia['Podiel'].round(1).astype(str)
        + " %)"
    )
else:
    df_vek_distribucia['Podiel'] = 0
    df_vek_distribucia['Text'] = pd.Series(dtype=str)

fig_vek_distribucia = px.bar(
    df_vek_distribucia,
    x='Vek_kat',
    y='Počet',
    text='Text',
    category_orders={'Vek_kat': poradie},
    color='Vek_kat',
    color_discrete_sequence=['#0F766E', '#14B8A6', '#2D728F', '#1D4E89', '#84A59D', '#3A7D44']
)
fig_vek_distribucia.update_traces(textposition='outside', showlegend=True)
apply_chart_theme(fig_vek_distribucia)
fig_vek_distribucia.update_xaxes(title_text="")
fig_vek_distribucia.update_layout(
    legend=dict(
        orientation='h',
        x=0.5,
        y=-0.24,
        xanchor='center',
        yanchor='top'
    ),
    margin=dict(l=24, r=24, t=28, b=96)
)
st.plotly_chart(fig_vek_distribucia, use_container_width=True)

# -----------------------------------------------
# VÝSLEDOK HOSPITALIZÁCIE PODĽA VEKU
# -----------------------------------------------
st.subheader("Výsledok hospitalizácie podľa veku")
df_vek = df_vysledok.groupby(['Vek_kat', 'Výsledok_hospitalizácie_popis']).size().reset_index(name='Počet')
total_vek = df_vek['Počet'].sum()
if total_vek > 0:
    df_vek['Podiel'] = (df_vek['Počet'] / total_vek) * 100
    df_vek['Text'] = (
        df_vek['Počet'].astype(int).astype(str)
        + " ("
        + df_vek['Podiel'].round(1).astype(str)
        + " %)"
    )
else:
    df_vek['Podiel'] = 0
    df_vek['Text'] = pd.Series(dtype=str)
fig_vek = px.bar(
    df_vek,
    x='Vek_kat',
    y='Počet',
    color='Výsledok_hospitalizácie_popis',
    text='Text',
    category_orders={'Vek_kat': poradie},
    color_discrete_map={
        '1 - Prepustenie domov': '#2ecc71',
        '2 - Presun na oddelenie': '#f39c12',
        '3 - Smrť': '#e74c3c'
    },
    barmode='group'
)
fig_vek.update_traces(textposition='outside')
apply_chart_theme(fig_vek)
fig_vek.update_xaxes(title_text="")
fig_vek.update_layout(
    legend=dict(
        orientation='h',
        x=0.5,
        y=-0.24,
        xanchor='center',
        yanchor='top',
        font=dict(color=PALETTE['text']),
        bgcolor=PALETTE['card']
    ),
    margin=dict(l=24, r=24, t=28, b=96)
)
st.plotly_chart(fig_vek, use_container_width=True)

# -----------------------------------------------
# DISTRIBÚCIA VLN COVID
# -----------------------------------------------
st.subheader("Vlna")
df_vlna_distribucia = df_filtered['vlna'].value_counts().reset_index()
df_vlna_distribucia.columns = ['vlna', 'Počet']
df_vlna_distribucia['vlna_label'] = df_vlna_distribucia['vlna'].astype(str)
total_vlna_distribucia = df_vlna_distribucia['Počet'].sum()
if total_vlna_distribucia > 0:
    df_vlna_distribucia['Podiel'] = (df_vlna_distribucia['Počet'] / total_vlna_distribucia) * 100
    df_vlna_distribucia['Text'] = (
        df_vlna_distribucia['Počet'].astype(int).astype(str)
        + " ("
        + df_vlna_distribucia['Podiel'].round(1).astype(str)
        + " %)"
    )
else:
    df_vlna_distribucia['Podiel'] = 0
    df_vlna_distribucia['Text'] = pd.Series(dtype=str)

vlna_options_str = sorted(
    df['vlna'].dropna().unique().tolist(),
    key=lambda v: (0, float(str(v))) if str(v).replace('.', '', 1).isdigit() else (1, str(v))
)

fig_vlna_distribucia = px.bar(
    df_vlna_distribucia,
    x='vlna_label',
    y='Počet',
    text='Text',
    category_orders={'vlna_label': [str(v) for v in vlna_options_str]},
    color='vlna_label',
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_vlna_distribucia.update_xaxes(type='category')
fig_vlna_distribucia.update_xaxes(title_text="")
fig_vlna_distribucia.update_traces(textposition='outside')
apply_chart_theme(fig_vlna_distribucia)
fig_vlna_distribucia.update_layout(
    legend=dict(
        orientation='h',
        x=0.5,
        y=-0.24,
        xanchor='center',
        yanchor='top'
    ),
    margin=dict(l=24, r=24, t=28, b=96)
)
st.plotly_chart(fig_vlna_distribucia, use_container_width=True)
