import os

import streamlit as st

from utils import (
    BASE_DIR,
    DRUG_OPTIONS,
    PALETTE,
    load_association_rules,
    load_data,
    parse_itemset_text,
    setup_page,
)

setup_page("Asociačné pravidlá")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] [data-testid="stButton"] > button {
        padding: 0 1rem !important;
        height: 2.4rem !important;
        min-height: 2.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Asociačné pravidlá")

ZAVAZNOST_LABELS = {
    1: "1 – Prepustenie domov",
    2: "2 – Presun na oddelenie",
    3: "3 – Smrť",
}

COMORBIDITY_ORDER = [
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Pečeňové ochorenia",
    "Onkologické ochorenia",
]

RULES_FILES = {
    "Apriori":   str(BASE_DIR / "asociacne_pravidla.xlsx"),
    "FP-Growth": str(BASE_DIR / "asociacne_pravidla_fpgrowth.xlsx"),
}


def antecedent_sort_key(item):
    
    if item in DRUG_OPTIONS:
        return (0, DRUG_OPTIONS.index(item), item)
    if item in COMORBIDITY_ORDER:
        return (1, COMORBIDITY_ORDER.index(item), item)
    if item.startswith("Veková kategória"):
        try:
            age_start = int(item.split()[-1].split("-")[0].split("+")[0])
        except (ValueError, IndexError):
            age_start = 999
        return (2, age_start, item)
    if item.startswith("Vlna"):
        try:
            wave_num = int(item.split("-")[-1].strip())
        except (ValueError, IndexError):
            wave_num = 999
        return (3, wave_num, item)
    if item.startswith("Pohlavie"):
        return (4, 0, item)
    return (0, 0, item)


def render_rules(rules_file_path, key_prefix):
    rules_file_name = os.path.basename(rules_file_path)
    rules_file_mtime = (
        os.path.getmtime(rules_file_path) if os.path.exists(rules_file_path) else None
    )
    rules_df = load_association_rules(rules_file_path, rules_file_mtime)

    if rules_df.empty:
        st.warning(f"Súbor {rules_file_name} sa nenašiel alebo je prázdny.")
        return

    required_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    missing_cols = [c for c in required_cols if c not in rules_df.columns]
    if missing_cols:
        st.error(f"V súbore {rules_file_name} chýbajú stĺpce: {', '.join(missing_cols)}")
        return

    rules_view = rules_df.copy()
    rules_view["antecedent_items"] = rules_view["antecedents"].apply(parse_itemset_text)
    rules_view["consequent_items"] = rules_view["consequents"].apply(parse_itemset_text)
    rules_view["antecedents_text"] = rules_view["antecedent_items"].apply(lambda x: ", ".join(x))
    rules_view["consequents_text"] = rules_view["consequent_items"].apply(lambda x: ", ".join(x))

    # --- Filtre ---
    filter_col1, filter_col2 = st.columns([2, 1])

    FILTER_LABEL_STYLE = "font-size:0.875rem; font-weight:600; margin-bottom:0.35rem; display:block;"

    with filter_col1:
        st.markdown(f"<span style='{FILTER_LABEL_STYLE}'>Predpoklad obsahuje</span>", unsafe_allow_html=True)
        all_antecedent_items = sorted(
            {item for row in rules_view["antecedent_items"] for item in row},
            key=antecedent_sort_key
        )
        selected_antecedents = st.multiselect(
            "Predpoklad obsahuje",
            options=all_antecedent_items,
            placeholder="Vybrať predpoklad...",
            label_visibility="collapsed",
            key=f"{key_prefix}_antecedents",
        )

    with filter_col2:
        st.markdown(f"<span style='{FILTER_LABEL_STYLE}'>Výsledok</span>", unsafe_allow_html=True)

        # Inicializácia — default: všetky tri zapnuté
        for val in [1, 2, 3]:
            key = f"{key_prefix}_sev_{val}"
            if key not in st.session_state:
                st.session_state[key] = True

        btn_cols = st.columns(3)
        for btn_col, val in zip(btn_cols, [1, 2, 3]):
            with btn_col:
                if st.button(
                    str(val),
                    key=f"{key_prefix}_sev_btn_{val}",
                    type="primary" if st.session_state[f"{key_prefix}_sev_{val}"] else "secondary",
                    use_container_width=True,
                ):
                    st.session_state[f"{key_prefix}_sev_{val}"] = not st.session_state[f"{key_prefix}_sev_{val}"]
                    st.rerun()

    # Dynamické maximá podľa datasetu
    max_support = rules_view["support"].max() if not rules_view.empty else 1.0
    max_confidence = rules_view["confidence"].max() if not rules_view.empty else 1.0
    max_lift = rules_view["lift"].max() if not rules_view.empty else 10.0
    
    # Defaults
    default_support = min(0.10, max_support)
    default_confidence = min(0.30, max_confidence)
    default_lift = min(1.0, max_lift)

    slider_col1, slider_col2, slider_col3 = st.columns(3)
    with slider_col1:
        min_support = st.slider("Min. Support", 0.10, max_support, default_support, 0.01, key=f"{key_prefix}_support")
    with slider_col2:
        min_confidence = st.slider("Min. Confidence", 0.30, max_confidence, default_confidence, 0.01, key=f"{key_prefix}_confidence")
    with slider_col3:
        min_lift = st.slider("Min. Lift", 1.0, max_lift, default_lift, 0.1, key=f"{key_prefix}_lift")

    # --- Filtrovanie ---
    filtered = rules_view[
        (rules_view["support"] >= min_support) &
        (rules_view["confidence"] >= min_confidence) &
        (rules_view["lift"] >= min_lift)
    ]

    active_severities = [val for val in [1, 2, 3] if st.session_state[f"{key_prefix}_sev_{val}"]]
    if active_severities:
        filtered = filtered[
            filtered["consequent_items"].apply(
                lambda items: any(
                    any(str(sev) in item for item in items)
                    for sev in active_severities
                )
            )
        ]
    else:
        filtered = filtered.iloc[0:0]  # nič nie je vybraté → prázdna tabuľka

    if selected_antecedents:
        filtered = filtered[
            filtered["antecedent_items"].apply(
                lambda items: all(item in items for item in selected_antecedents)
            )
        ]

    if not filtered.empty:
        filtered = filtered.sort_values("lift", ascending=False)

    # --- Metriky ---
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Počet pravidiel", len(filtered))
    with m2:
        st.metric("Top Lift", f"{filtered['lift'].max():.3f}" if len(filtered) > 0 else "0.000")

    # --- Tabuľka ---
    if len(filtered) > 0:
        table = filtered[[
            "antecedents_text", "consequents_text", "support", "confidence", "lift"
        ]].rename(columns={
            "antecedents_text": "Predpoklad",
            "consequents_text": "Výsledok",
            "support":          "Support",
            "confidence":       "Confidence",
            "lift":             "Lift",
        })

        table_styled = table.style.set_properties(**{
            "background-color": PALETTE["card"],
            "color":            PALETTE["text"],
        }).set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", PALETTE["card"]),
                ("color",            PALETTE["text"]),
                ("font-weight",      "600"),
            ],
        }])

        st.dataframe(table_styled, use_container_width=True, hide_index=True)
    else:
        st.info("Žiadne pravidlá nespĺňajú zvolené kritéria.")


# --- Záložky Apriori / FP-Growth ---
tab_apriori, tab_fpgrowth = st.tabs(["Apriori", "FP-Growth"])

with tab_apriori:
    render_rules(RULES_FILES["Apriori"], "apriori")

with tab_fpgrowth:
    render_rules(RULES_FILES["FP-Growth"], "fpgrowth")
