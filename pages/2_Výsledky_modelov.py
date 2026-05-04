import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    PALETTE,
    apply_chart_theme,
    compute_model_results,
    load_data,
    setup_page,
)

setup_page("Výsledky modelov")

st.title("Výsledky modelov")

df = load_data()
results = compute_model_results(df)
labels = results["labels"]

MODEL_NAMES = ["Random Forest", "XGBoost", "Logistická regresia", "Baseline"]

# -----------------------------------------------
# POROVNÁVACÍ BAR CHART — Accuracy, Balanced Accuracy, Macro F1
# -----------------------------------------------
st.subheader("Porovnanie modelov")

fig_compare = go.Figure()

fig_compare.add_trace(go.Bar(
    name="Accuracy (%)",
    x=MODEL_NAMES,
    y=[results[m]["acc"] * 100 for m in MODEL_NAMES],
    marker_color=PALETTE["accent"],
    text=[f"{results[m]['acc'] * 100:.2f} %" for m in MODEL_NAMES],
    textposition="outside",
))

fig_compare.add_trace(go.Bar(
    name="Balanced Accuracy (%)",
    x=MODEL_NAMES,
    y=[results[m]["balanced_acc"] * 100 for m in MODEL_NAMES],
    marker_color=PALETTE["accent_light"],
    text=[f"{results[m]['balanced_acc'] * 100:.2f} %" for m in MODEL_NAMES],
    textposition="outside",
))

fig_compare.add_trace(go.Bar(
    name="Macro F1",
    x=MODEL_NAMES,
    y=[results[m]["f1"] * 100 for m in MODEL_NAMES],
    marker_color="#2D728F",
    text=[f"{results[m]['f1']:.3f}" for m in MODEL_NAMES],
    textposition="outside",
))

fig_compare.update_layout(
    barmode="group",
    yaxis_title="Hodnota (%)",
    legend=dict(orientation="h", x=0.5, y=-0.18, xanchor="center"),
    margin=dict(l=24, r=24, t=28, b=80),
)
apply_chart_theme(fig_compare)
st.plotly_chart(fig_compare, use_container_width=True)

# -----------------------------------------------
# METRIKY — karty
# -----------------------------------------------
st.markdown("---")
st.subheader("Detailné metriky")

mcols = st.columns(len(MODEL_NAMES))
for col, model in zip(mcols, MODEL_NAMES):
    with col:
        st.markdown(f"**{model}**")
        st.metric("Accuracy", f"{results[model]['acc'] * 100:.2f} %")
        st.metric("Balanced Accuracy", f"{results[model]['balanced_acc'] * 100:.2f} %")
        st.metric("Macro F1", f"{results[model]['f1']:.3f}")

# -----------------------------------------------
# ROZLOŽENIE TRIED V DATASETE
# -----------------------------------------------
st.markdown("---")
st.subheader("Rozloženie tried v datasete")

zavaznost_map = {1: "1 – Prepustenie domov", 2: "2 – Presun na oddelenie", 3: "3 – Smrť"}
class_counts = df["Závažnosť priebehu ochorenia"].value_counts().sort_index()
total = class_counts.sum()

dist_cols = st.columns(3)
for col, (cls, count) in zip(dist_cols, class_counts.items()):
    with col:
        st.metric(
            label=zavaznost_map.get(cls, str(cls)),
            value=f"{count}",
            delta=f"{count / total * 100:.1f} % z celku",
            delta_color="off",
        )

# -----------------------------------------------
# CONFUSION MATRICES — 2×2 mriežka
# -----------------------------------------------
st.markdown("---")
st.subheader("Confusion matice")

cm_color_scale = ['#F6FBFF', '#E6F4FA', '#CBE8F3', '#A8D7EB', '#84C5E2']

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

grid = [
    (row1_col1, "Random Forest"),
    (row1_col2, "XGBoost"),
    (row2_col1, "Logistická regresia"),
    (row2_col2, "Baseline"),
]

for col, model in grid:
    with col:
        st.markdown(f"**{model}**")
        cm_fig = px.imshow(
            results[model]["cm"],
            labels=dict(x="Predikovaná trieda", y="Skutočná trieda", color="Počet"),
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=cm_color_scale,
        )
        apply_chart_theme(cm_fig)
        cm_fig.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color=PALETTE["text"]),
                title=dict(text="Počet", font=dict(color=PALETTE["text"]))
            )
        )
        cm_fig.update_xaxes(
            tickfont=dict(color=PALETTE["text"]),
            title_font=dict(color=PALETTE["text"]),
            color=PALETTE["text"],
        )
        cm_fig.update_yaxes(
            tickfont=dict(color=PALETTE["text"]),
            title_font=dict(color=PALETTE["text"]),
            color=PALETTE["text"],
        )
        cm_fig.update_traces(textfont=dict(color=PALETTE["text"]))
        st.plotly_chart(cm_fig, use_container_width=True)
