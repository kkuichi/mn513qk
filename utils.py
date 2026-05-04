import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent

PALETTE = {
    'bg': "#F8F4F4",
    'card': '#FFFFFF',
    'text': '#12343B',
    'muted': '#5E7C82',
    'accent': '#0F766E',
    'accent_light': '#14B8A6'
}

VYSLEDOK_COL = 'Závažnosť priebehu ochorenia'

VYSLEDOK_LABELS = {
    1: '1 - Prepustenie domov',
    2: '2 - Presun na oddelenie',
    3: '3 - Smrť'
}

COMORBIDITY_OPTIONS = [
    'Hypertenzia',
    'Diabetes mellitus',
    'Kardiovaskulárne ochorenia',
    'Chronické respiračné ochorenia',
    'Renálne ochorenia',
    'Pečeňové ochorenia',
    'Onkologické ochorenia'
]

DRUG_OPTIONS = [
    'MD652 | FABIFLU TABLETS',
    'MD656 IV-BECT 6MG (ivermectin)',
    '5042D | VEKLURY',
    '9547D | PAXLOVID',
    'LAGEVRIO',
    '00584 | PYRIDOXIN LÉČIVA INJ',
    'Vitamin C',
    'Vitamin D',
    '00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK',
    '00449 | EREVIT 300 MG/ML',
    'Prednison',
    'Dexametazon',
    '2410B HYDROCORTISONE',
    '3242C | OLUMIANT 4 MG',
    'Kineret',
    'RoActemra',
    '34045 | POLYOXIDONIUM 6 MG',
    '87299 | IMUNOR',
    '56930 IMMODIN',
    'Isoprinosine/INOMED',
    '35715 Azithromycin',
    '45954 Ceftriaxon',
    'Moxifloxacin',
    'Ciprofloxacin',
    'PPI',
    '94918 AMBROBENE',
    '24859 PENTOXYPHILLINUM',
    '8893 ACC INJEKT',
    '24949 CODEIN ',
    '26846 OXANTIL',
    'Antikoagulancia',
    'Antiagregacne'
]


def _build_css():
    return f"""
    <style>
    .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
    [data-testid="stAppViewContainer"] :is(p,label,span,div,h1,h2,h3) {{ color: {PALETTE['text']} !important; }}

    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0F766E 0%, #155E75 100%); }}
    [data-testid="stSidebar"] :is(p,label,span,div,h1,h2,h3) {{ color: #E6FFFB !important; }}

    [data-testid="stHeader"] {{ background: linear-gradient(90deg, #0F766E 0%, #155E75 100%); }}
    [data-testid="stHeader"] :is(*,button,a) {{ color: #FFFFFF !important; }}

    [data-testid="collapsedControl"] button svg {{ color: #FFFFFF !important; }}

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {{ background-color: {PALETTE['card']} !important; border-color: #DCEBE9 !important; }}
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] :is(span,input,[role="combobox"],svg,[data-baseweb="tag"] span) {{ color: {PALETTE['text']} !important; }}
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] input {{ -webkit-text-fill-color: {PALETTE['text']} !important; }}
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] input::placeholder {{ color: {PALETTE['muted']} !important; opacity: 1 !important; }}
    [data-testid="stAppViewContainer"] div[data-baseweb="popover"] :is(ul,li,div) {{ color: {PALETTE['text']} !important; background-color: {PALETTE['card']} !important; }}

    [data-testid="stDataFrame"], [data-testid="stTable"] {{ background-color: {PALETTE['card']} !important; color: {PALETTE['text']} !important; }}
    [data-testid="stDataFrame"] :is([role="columnheader"],[role="gridcell"],[role="rowheader"],[role="columnheader"] *,[role="gridcell"] *,[role="rowheader"] *),
    [data-testid="stTable"] :is(th,td) {{ background-color: {PALETTE['card']} !important; color: {PALETTE['text']} !important; }}
    [data-testid="stDataFrame"] [role="columnheader"] {{ font-weight: 600; border-bottom: 1px solid #DCEBE9 !important; }}

    [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        border-bottom: 1px solid #DCEBE9;
        margin-bottom: 0.75rem;
    }}
    [data-baseweb="tab"] {{
        background-color: #EAF4F3 !important;
        color: {PALETTE['text']} !important;
        border: 1px solid #CDE2DF !important;
        border-bottom: none !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 0.45rem 0.9rem !important;
    }}
    [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #FECACA !important;
        color: #7F1D1D !important;
        border-color: #FCA5A5 !important;
    }}
    [data-baseweb="tab"] p {{
        color: inherit !important;
        font-weight: 600 !important;
    }}

    [data-testid="stButton"] > button {{
        background: #F2F7F6 !important;
        color: {PALETTE['text']} !important;
        border: 1px solid #D5E6E3 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.15s ease-in-out !important;
    }}
    [data-testid="stButton"] > button:hover {{
        background: #E5F2EF !important;
        border-color: {PALETTE['accent']} !important;
        color: {PALETTE['text']} !important;
    }}
    [data-testid="stButton"] > button:focus-visible {{
        outline: 2px solid {PALETTE['accent_light']} !important;
        outline-offset: 2px !important;
    }}
    [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
        background: #14B8A6 !important;
        color: #FFFFFF !important;
        border-color: #14B8A6 !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover,
    [data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {{
        background: #0FA392 !important;
        border-color: #0FA392 !important;
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] [data-testid="stButton"] > button {{
        background: rgba(255, 255, 255, 0.14) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.28) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
        background: rgba(255, 255, 255, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
        background: #E6FFFB !important;
        color: {PALETTE['accent']} !important;
        border-color: #E6FFFB !important;
    }}
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {{
        background: #CFF7F1 !important;
        color: {PALETTE['accent']} !important;
        border-color: #CFF7F1 !important;
    }}

    /* Toolbar buttons in dataframe tables */
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button svg,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button svg * {{
        color: {PALETTE['accent_light']} !important;
    }}
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover svg,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover svg * {{
        color: {PALETTE['accent']} !important;
    }}

    /* Posun obsahu nižšie, aby neprekryl horný header */
    [data-testid="stAppViewContainer"] .block-container {{
        padding-top: 4rem !important;
    }}
    [data-testid="stAppViewContainer"] h1 {{
        margin-top: 0.2rem !important;
        padding-top: 0 !important;
    }}
    </style>
    """


def setup_page(page_title="COVID-19 Dashboard"):
    st.set_page_config(page_title=page_title, layout="wide")
    st.markdown(_build_css(), unsafe_allow_html=True)


def vek_sort_key(value):
    text = str(value).strip()
    match_range = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
    if match_range:
        return (0, int(match_range.group(1)), int(match_range.group(2)), text)

    match_plus = re.match(r"^(\d+)\s*\+$", text)
    if match_plus:
        start = int(match_plus.group(1))
        return (0, start, 999, text)

    return (1, 9999, 9999, text)


def vlna_sort_key(value):
    if pd.isna(value):
        return (1, "")
    text = str(value).strip()
    match_number = re.match(r"^\d+(?:\.\d+)?$", text)
    if match_number:
        return (0, float(text))
    return (1, text)


def apply_chart_theme(fig):
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor=PALETTE['card'],
        plot_bgcolor=PALETTE['card'],
        font=dict(color=PALETTE['text']),
        title_font=dict(color=PALETTE['text']),
        legend=dict(
            font=dict(color=PALETTE['text']),
            title=dict(text=''),
            tracegroupgap=0,
        ),
        legend_title_font=dict(color=PALETTE['text']),
        hoverlabel=dict(
            font=dict(color=PALETTE['text']),
            bgcolor=PALETTE['card'],
            bordercolor="#EBDCDC",
        ),
        title=dict(text=fig.layout.title.text or ''),
    )
    fig.update_xaxes(
        gridcolor='#DCEBE9',
        zerolinecolor='#DCEBE9',
        tickfont=dict(color=PALETTE['text']),
        title_text=(fig.layout.xaxis.title.text or ''),
    )
    fig.update_yaxes(
        gridcolor='#DCEBE9',
        zerolinecolor='#DCEBE9',
        tickfont=dict(color=PALETTE['text']),
        title_text=(fig.layout.yaxis.title.text or ''),
    )
    fig.update_traces(textfont=dict(color=PALETTE['text']))


def parse_itemset_text(value):
    if pd.isna(value):
        return []
    return re.findall(r"'([^']+)'", str(value))


def set_checkbox_group(keys, value):
    for key in keys:
        st.session_state[key] = value


@st.cache_data
def load_data():
    return pd.read_excel(BASE_DIR / "Upraveny_dataset.xlsx")


@st.cache_data
def load_association_rules(file_path, file_mtime):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def compute_model_results(source_df):
    work_df = source_df.copy()

    work_df["Pohlavie_num"] = work_df["Pohlavie"].map({"Muž": 0, "Žena": 1})
    work_df["Vek_kat_num"] = work_df["Vek_kat"].astype("category").cat.codes
    work_df["vlna_num"] = work_df["vlna"].astype("category").cat.codes

    feature_cols_num = [
        "Pohlavie_num", "Vek_kat_num", "vlna_num",
        "Hypertenzia", "Diabetes mellitus", "Kardiovaskulárne ochorenia",
        "Chronické respiračné ochorenia", "Renálne ochorenia",
        "Pečeňové ochorenia", "Onkologické ochorenia",
        "MD652 | FABIFLU TABLETS", "MD656 IV-BECT 6MG (ivermectin)",
        "5042D | VEKLURY", "9547D | PAXLOVID", "LAGEVRIO",
        "00584 | PYRIDOXIN LÉČIVA INJ", "Vitamin C", "Vitamin D",
        "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
        "00449 | EREVIT 300 MG/ML", "Prednison", "Dexametazon",
        "2410B HYDROCORTISONE", "3242C | OLUMIANT 4 MG",
        "Kineret", "RoActemra", "34045 | POLYOXIDONIUM 6 MG",
        "87299 | IMUNOR", "56930 IMMODIN", "Isoprinosine/INOMED",
        "35715 Azithromycin", "45954 Ceftriaxon",
        "Moxifloxacin", "Ciprofloxacin", "PPI",
        "94918 AMBROBENE", "24859 PENTOXYPHILLINUM",
        "8893 ACC INJEKT", "24949 CODEIN ", "26846 OXANTIL",
        "Antikoagulancia", "Antiagregacne"
    ]

    X = work_df[feature_cols_num]
    y = work_df["Závažnosť priebehu ochorenia"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    labels = [1, 2, 3]

    # Random Forest
    rf = RandomForestClassifier(
        # Avoid loky/joblib cleanup warnings on Windows Streamlit reruns.
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # XGBoost — vyžaduje 0-indexované triedy (1,2,3 → 0,1,2)
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        objective='multi:softmax',
        eval_metric='mlogloss',
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train - 1)
    y_pred_xgb = xgb.predict(X_test) + 1  # späť na 1,2,3

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Baseline (Dummy)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "acc":          accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "f1":           f1_score(y_true, y_pred, average='macro', zero_division=0),
            "cm":           confusion_matrix(y_true, y_pred, labels=labels),
        }

    feature_importance = (
        pd.DataFrame({
            "feature":    feature_cols_num,
            "importance": rf.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "labels":             labels,
        "feature_importance": feature_importance,
        "Random Forest":      metrics(y_test, y_pred_rf),
        "XGBoost":            metrics(y_test, y_pred_xgb),
        "Logistická regresia": metrics(y_test, y_pred_lr),
        "Baseline":           metrics(y_test, y_pred_dummy),
    }


def render_association_rules_section(section_title, rules_file_path, key_prefix):
    st.markdown(f"### {section_title}")

    rules_file_name = os.path.basename(rules_file_path)
    rules_file_mtime = os.path.getmtime(rules_file_path) if os.path.exists(rules_file_path) else None
    rules_df = load_association_rules(rules_file_path, rules_file_mtime)

    if rules_df.empty:
        st.warning(f"Súbor {rules_file_name} sa nenašiel alebo je prázdny.")
        return

    required_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    missing_cols = [col for col in required_cols if col not in rules_df.columns]

    if missing_cols:
        st.error(f"V súbore {rules_file_name} chýbajú stĺpce: {', '.join(missing_cols)}")
        return

    rules_view = rules_df.copy()
    rules_view['antecedent_items'] = rules_view['antecedents'].apply(parse_itemset_text)
    rules_view['consequent_items'] = rules_view['consequents'].apply(parse_itemset_text)
    rules_view['antecedents_text'] = rules_view['antecedent_items'].apply(lambda x: ', '.join(x))
    rules_view['consequents_text'] = rules_view['consequent_items'].apply(lambda x: ', '.join(x))

    all_antecedent_items = sorted({item for row in rules_view['antecedent_items'] for item in row})
    all_consequent_items = sorted({item for row in rules_view['consequent_items'] for item in row})

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_antecedents = st.multiselect(
            "Antecedent obsahuje",
            options=all_antecedent_items,
            key=f"{key_prefix}_antecedents"
        )
    with f2:
        selected_consequents = st.multiselect(
            "Consequent obsahuje",
            options=all_consequent_items,
            key=f"{key_prefix}_consequents"
        )
    with f3:
        rules_sort = st.selectbox(
            "Zoradiť podľa",
            options=['lift', 'confidence', 'support'],
            index=0,
            key=f"{key_prefix}_sort"
        )

    m1, m2, m3 = st.columns(3)
    with m1:
        min_support = st.slider("Min support", 0.0, 1.0, 0.02, 0.01, key=f"{key_prefix}_support")
    with m2:
        min_confidence = st.slider("Min confidence", 0.0, 1.0, 0.30, 0.01, key=f"{key_prefix}_confidence")
    with m3:
        min_lift = st.slider("Min lift", 0.0, 10.0, 1.0, 0.1, key=f"{key_prefix}_lift")

    filtered_rules = rules_view[
        (rules_view['support'] >= min_support) &
        (rules_view['confidence'] >= min_confidence) &
        (rules_view['lift'] >= min_lift)
    ]

    if selected_antecedents:
        filtered_rules = filtered_rules[
            filtered_rules['antecedent_items'].apply(lambda items: all(item in items for item in selected_antecedents))
        ]

    if selected_consequents:
        filtered_rules = filtered_rules[
            filtered_rules['consequent_items'].apply(lambda items: all(item in items for item in selected_consequents))
        ]

    filtered_rules = filtered_rules.sort_values(rules_sort, ascending=False)

    k1, k2 = st.columns(2)
    with k1:
        st.metric("Počet pravidiel", len(filtered_rules))
    with k2:
        if len(filtered_rules) > 0:
            st.metric("Top lift", f"{filtered_rules['lift'].max():.2f}")
        else:
            st.metric("Top lift", "0.00")

    rules_table = filtered_rules[[
        'antecedents_text',
        'consequents_text',
        'support',
        'confidence',
        'lift'
    ]].rename(columns={
        'antecedents_text': 'Antecedent',
        'consequents_text': 'Consequent',
        'support': 'Support',
        'confidence': 'Confidence',
        'lift': 'Lift'
    })

    rules_table_styled = rules_table.style.set_properties(**{
        'background-color': PALETTE['card'],
        'color': PALETTE['text']
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', PALETTE['card']),
                ('color', PALETTE['text']),
                ('font-weight', '600')
            ]
        }
    ])

    st.dataframe(
        rules_table_styled,
        use_container_width=True,
        hide_index=True
    )


def render_sidebar_filters(df):
    """Vykresli filtre v sidebare a vrati (df_filtered, vek_options)."""
    st.sidebar.header("Filtre")

    # Vekové kategórie
    vek_options = sorted(df['Vek_kat'].dropna().unique().tolist(), key=vek_sort_key)
    st.sidebar.markdown("**Veková kategória**")
    vek_btn_col1, vek_btn_col2 = st.sidebar.columns(2)
    vek_keys = [str(vek) for vek in vek_options]
    with vek_btn_col1:
        st.button("Vybrať všetky", key="vek_select_all", use_container_width=True, on_click=set_checkbox_group, args=(vek_keys, True))
    with vek_btn_col2:
        st.button("Zrušiť všetky", key="vek_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(vek_keys, False))
    vek_selected = {}
    for vek in vek_options:
        vek_key = str(vek)
        if vek_key not in st.session_state:
            st.session_state[vek_key] = True
        vek_selected[vek] = st.sidebar.checkbox(vek_key, key=vek_key)
    vek_vyber = [value for value, selected in vek_selected.items() if selected]

    # Pohlavie
    pohlavie_options = df['Pohlavie'].dropna().unique().tolist()
    st.sidebar.markdown("**Pohlavie**")
    pohlavie_btn_col1, pohlavie_btn_col2 = st.sidebar.columns(2)
    pohlavie_keys = [str(pohlavie) for pohlavie in pohlavie_options]
    with pohlavie_btn_col1:
        st.button("Vybrať všetky", key="pohlavie_select_all", use_container_width=True, on_click=set_checkbox_group, args=(pohlavie_keys, True))
    with pohlavie_btn_col2:
        st.button("Zrušiť všetky", key="pohlavie_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(pohlavie_keys, False))
    pohlavie_selected = {}
    for pohlavie in pohlavie_options:
        pohlavie_key = str(pohlavie)
        if pohlavie_key not in st.session_state:
            st.session_state[pohlavie_key] = True
        pohlavie_selected[pohlavie] = st.sidebar.checkbox(pohlavie_key, key=pohlavie_key)
    pohlavie_vyber = [value for value, selected in pohlavie_selected.items() if selected]

    # Vlna
    vlna_options = sorted(df['vlna'].dropna().unique().tolist(), key=vlna_sort_key)
    st.sidebar.markdown("**Vlna COVID**")
    vlna_btn_col1, vlna_btn_col2 = st.sidebar.columns(2)
    vlna_keys = [str(vlna) for vlna in vlna_options]
    with vlna_btn_col1:
        st.button("Vybrať všetky", key="vlna_select_all", use_container_width=True, on_click=set_checkbox_group, args=(vlna_keys, True))
    with vlna_btn_col2:
        st.button("Zrušiť všetky", key="vlna_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(vlna_keys, False))
    vlna_selected = {}
    for vlna in vlna_options:
        vlna_key = str(vlna)
        if vlna_key not in st.session_state:
            st.session_state[vlna_key] = True
        vlna_selected[vlna] = st.sidebar.checkbox(vlna_key, key=vlna_key)
    vlna_vyber = [value for value, selected in vlna_selected.items() if selected]

    # Výsledok hospitalizácie
    vysledok_options = sorted(df[VYSLEDOK_COL].dropna().unique().tolist())
    st.sidebar.markdown("**Výsledok hospitalizácie**")
    vysledok_btn_col1, vysledok_btn_col2 = st.sidebar.columns(2)
    vysledok_keys = [VYSLEDOK_LABELS.get(vysledok, str(vysledok)) for vysledok in vysledok_options]
    with vysledok_btn_col1:
        st.button("Vybrať všetky", key="vysledok_select_all", use_container_width=True, on_click=set_checkbox_group, args=(vysledok_keys, True))
    with vysledok_btn_col2:
        st.button("Zrušiť všetky", key="vysledok_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(vysledok_keys, False))
    vysledok_selected = {}
    for vysledok in vysledok_options:
        label = VYSLEDOK_LABELS.get(vysledok, str(vysledok))
        if label not in st.session_state:
            st.session_state[label] = True
        vysledok_selected[vysledok] = st.sidebar.checkbox(label, key=label)
    vysledok_vyber = [value for value, selected in vysledok_selected.items() if selected]

    # Komorbidity
    st.sidebar.markdown("**Komorbidity**")
    comorbidity_logic = st.sidebar.radio(
        "Logika komorbidít",
        options=["AND", "OR"],
        index=0,
        help="AND = pacient musí mať všetky vybrané komorbidity, OR = stačí aspoň jedna vybraná komorbidita."
    )

    comorbidity_btn_col1, comorbidity_btn_col2 = st.sidebar.columns(2)
    comorbidity_keys = [f"comorbidity_{idx}" for idx, _ in enumerate(COMORBIDITY_OPTIONS)]
    with comorbidity_btn_col1:
        st.button("Vybrať všetky", key="comorbidity_select_all", use_container_width=True, on_click=set_checkbox_group, args=(comorbidity_keys, True))
    with comorbidity_btn_col2:
        st.button("Zrušiť všetky", key="comorbidity_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(comorbidity_keys, False))

    comorbidity_selected = {}
    for idx, comorbidity in enumerate(COMORBIDITY_OPTIONS):
        key = f"comorbidity_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        comorbidity_selected[comorbidity] = st.sidebar.checkbox(comorbidity, key=key)

    # Lieky
    st.sidebar.markdown("**Lieky**")
    drug_logic = st.sidebar.radio(
        "Logika liekov",
        options=["AND", "OR"],
        index=0,
        help="AND = pacient musí mať všetky vybrané lieky, OR = stačí aspoň jeden vybraný liek."
    )

    drug_btn_col1, drug_btn_col2 = st.sidebar.columns(2)
    drug_keys = [f"drug_{idx}" for idx, _ in enumerate(DRUG_OPTIONS)]
    with drug_btn_col1:
        st.button("Vybrať všetky", key="drug_select_all", use_container_width=True, on_click=set_checkbox_group, args=(drug_keys, True))
    with drug_btn_col2:
        st.button("Zrušiť všetky", key="drug_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(drug_keys, False))

    drug_selected = {}
    for idx, drug_name in enumerate(DRUG_OPTIONS):
        key = f"drug_{idx}"
        if key not in st.session_state:
            st.session_state[key] = False
        drug_selected[drug_name] = st.sidebar.checkbox(drug_name, key=key)

    # Filtrovanie
    df_filtered = df[
        (df['Vek_kat'].isin(vek_vyber)) &
        (df['Pohlavie'].isin(pohlavie_vyber)) &
        (df['vlna'].isin(vlna_vyber)) &
        (df[VYSLEDOK_COL].isin(vysledok_vyber))
    ]

    selected_comorbidities = [c for c, selected in comorbidity_selected.items() if selected]
    valid_comorbidities = [c for c in selected_comorbidities if c in df_filtered.columns]
    if valid_comorbidities:
        if comorbidity_logic == "AND":
            mask = df_filtered[valid_comorbidities].eq(True).all(axis=1)
        else:
            mask = df_filtered[valid_comorbidities].eq(True).any(axis=1)
        df_filtered = df_filtered[mask]

    selected_drugs = [d for d, selected in drug_selected.items() if selected]
    valid_drugs = [d for d in selected_drugs if d in df_filtered.columns]
    if valid_drugs:
        if drug_logic == "AND":
            drug_mask = df_filtered[valid_drugs].eq(True).all(axis=1)
        else:
            drug_mask = df_filtered[valid_drugs].eq(True).any(axis=1)
        df_filtered = df_filtered[drug_mask]

    return df_filtered, vek_options
