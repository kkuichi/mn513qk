import pandas as pd
import streamlit as st

from utils import COMORBIDITY_OPTIONS, load_data, setup_page

setup_page("Prevalencia komorbidít")

st.title("Prevalencia komorbidít")

df = load_data()

# Prevalencia v bežnej populácii — placeholder hodnoty, doplň skutočné
POPULATION_PREVALENCE = {
    "Hypertenzia":                    "33,0 %",
    "Diabetes mellitus":              "10,5 %",
    "Kardiovaskulárne ochorenia":     "6,8 %",
    "Chronické respiračné ochorenia": "10,3 %",
    "Renálne ochorenia":              "9,1 %",
    "Pečeňové ochorenia":             "29,8 %",
    "Onkologické ochorenia":          "4,6 %",
}

# Zdroje — doplň skutočný text a URL
SOURCES = {
    "Hypertenzia":                    "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC8446938/' target='_blank'>PMC8446938</a>",
    "Diabetes mellitus":              "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC11057359/' target='_blank'>PMC11057359</a>",
    "Kardiovaskulárne ochorenia":     "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC7755038/' target='_blank'>PMC7755038</a>",
    "Chronické respiračné ochorenia": "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC9050565/' target='_blank'>PMC9050565</a>",
    "Renálne ochorenia":              "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC7049905/' target='_blank'>PMC7049905</a>",
    "Pečeňové ochorenia":             "<a href='https://doi.org/10.1016/j.cgh.2021.12.002' target='_blank'>10.1016/j.cgh.2021.12.002</a>",
    "Onkologické ochorenia":          "<a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC5807846/' target='_blank'>PMC5807846</a>",
}

if len(df) == 0:
    st.warning("Nie sú dostupné dáta na výpočet prevalencie.")
else:
    prevalence_cols = [col for col in COMORBIDITY_OPTIONS if col in df.columns]

    if not prevalence_cols:
        st.warning("V datasete sa nenašli požadované stĺpce pre prevalenciu.")
    else:
        dataset_population = len(df)
        dataset_positive = df[prevalence_cols].eq(True).sum()
        if dataset_population > 0:
            fixed_dataset_prevalence = (dataset_positive / dataset_population * 100).fillna(0).round(2)
        else:
            fixed_dataset_prevalence = pd.Series(0.0, index=prevalence_cols)

        prevalence_table = pd.DataFrame({
            "Komorbidita": prevalence_cols,
            "Nameraná prevalencia (%)": fixed_dataset_prevalence.reindex(prevalence_cols).values
        }).sort_values("Nameraná prevalencia (%)", ascending=False).reset_index(drop=True)

        # Čísla zdrojov podľa poradia zobrazenia (1 = prvý riadok zhora)
        ref_order = {row["Komorbidita"]: idx + 1 for idx, row in prevalence_table.iterrows()}

        rows_html = []
        for _, row in prevalence_table.iterrows():
            komorbidita = row["Komorbidita"]
            dataset_val = f"{row['Nameraná prevalencia (%)']:.2f} %"
            ref_num = ref_order[komorbidita]

            pop_val = POPULATION_PREVALENCE.get(komorbidita, "")
            pop_cell = f'{pop_val} <sup>{ref_num}</sup>' if pop_val else ""

            rows_html.append(
                "<div style='display:grid;grid-template-columns:2fr 1.5fr 1.5fr;gap:1rem;padding:0.15rem 0;'>"
                f"<div>{komorbidita}</div>"
                f"<div>{dataset_val}</div>"
                f"<div>{pop_cell}</div>"
                "</div>"
            )

        st.markdown(
            f"""
            <div style="font-family: inherit; font-size: 1rem;">
                <div style="display:grid;grid-template-columns:2fr 1.5fr 1.5fr;gap:1rem;font-weight:600;padding-bottom:0.25rem;border-bottom:1px solid #DCEBE9;margin-bottom:0.4rem;">
                    <div>Komorbidita</div>
                    <div>Nameraná prevalencia v datasete</div>
                    <div>Prevalencia v bežnej populácii</div>
                </div>
                {''.join(rows_html)}
            </div>
            """,
            unsafe_allow_html=True
        )

        # -----------------------------------------------
        # ZDROJE
        # -----------------------------------------------
        st.markdown("---")
        st.markdown("<h4 style='color:#12343B;'>Zdroje</h4>", unsafe_allow_html=True)

        sources_html = []
        for _, row in prevalence_table.iterrows():
            komorbidita = row["Komorbidita"]
            ref_num = ref_order[komorbidita]
            source_text = SOURCES.get(komorbidita, "[Zdroj]")
            sources_html.append(
                f'<div style="padding:0.15rem 0; font-size:0.9rem;">'
                f'{ref_num}. {source_text}'
                f'</div>'
            )

        st.markdown(
            f'<div style="font-family: inherit;">{"".join(sources_html)}</div>',
            unsafe_allow_html=True
        )
