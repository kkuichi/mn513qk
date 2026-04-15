import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# streamlit run "c:/Users/Martin Novy/Documents/SKOLA/Bakalarka/Test Environment/dashboard.py"

# -----------------------------------------------
# NASTAVENIE STRÁNKY
# -----------------------------------------------
st.set_page_config(
    page_title="COVID-19 Dashboard",
    layout="wide"
)

PALETTE = {
    'bg': "#F8F4F4",
    'card': '#FFFFFF',
    'text': '#12343B',
    'muted': '#5E7C82',
    'accent': '#0F766E',
    'accent_light': '#14B8A6'
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PALETTE['bg']}; color: {PALETTE['text']}; }}
    [data-testid="stAppViewContainer"] :is(p,label,span,div,h1,h2,h3) {{ color: {PALETTE['text']} !important; }}

    [data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0F766E 0%, #155E75 100%); }}
    [data-testid="stSidebar"] :is(p,label,span,div,h1,h2,h3) {{ color: #E6FFFB !important; }}

    [data-testid="stHeader"] {{ background: linear-gradient(90deg, #0F766E 0%, #155E75 100%); }}
    [data-testid="stHeader"] :is(*,button,a) {{ color: #FFFFFF !important; }}

    div[data-baseweb="select"] > div {{ background-color: {PALETTE['card']} !important; border-color: #DCEBE9 !important; }}
    div[data-baseweb="select"] :is(span,input,[role="combobox"],svg,[data-baseweb="tag"] span) {{ color: {PALETTE['text']} !important; }}
    div[data-baseweb="select"] input {{ -webkit-text-fill-color: {PALETTE['text']} !important; }}
    div[data-baseweb="select"] input::placeholder {{ color: {PALETTE['muted']} !important; opacity: 1 !important; }}
    div[data-baseweb="popover"] :is(ul,li,div) {{ color: {PALETTE['text']} !important; background-color: {PALETTE['card']} !important; }}

    [data-testid="stDataFrame"], [data-testid="stTable"] {{ background-color: {PALETTE['card']} !important; color: {PALETTE['text']} !important; }}
    [data-testid="stDataFrame"] :is([role="columnheader"],[role="gridcell"],[role="rowheader"],[role="columnheader"] *,[role="gridcell"] *,[role="rowheader"] *),
    [data-testid="stTable"] :is(th,td) {{ background-color: {PALETTE['card']} !important; color: {PALETTE['text']} !important; }}
    [data-testid="stDataFrame"] [role="columnheader"] {{ font-weight: 600; border-bottom: 1px solid #DCEBE9 !important; }}

    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button svg,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button svg * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover svg,
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover svg * {{
        color: #FFFFFF !important;
    }}
    [data-testid="stDataFrame"] [data-testid="stElementToolbar"] button:hover {{
        background-color: transparent !important;
    }}

    /* Explicitny styl pre taby, aby neboli skryte globalnym themingom */
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
        background-color: {PALETTE['accent']} !important;
        color: #FFFFFF !important;
        border-color: {PALETTE['accent']} !important;
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
        background: {PALETTE['accent']} !important;
        color: #FFFFFF !important;
        border-color: {PALETTE['accent']} !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover,
    [data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover {{
        background: #0B5F58 !important;
        border-color: #0B5F58 !important;
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
    </style>
    """,
    unsafe_allow_html=True
)

if "top_menu_section" not in st.session_state:
    st.session_state["top_menu_section"] = "Prehľad"

def set_top_menu_section(section_name):
    st.session_state["top_menu_section"] = section_name

current_menu_section = st.session_state["top_menu_section"]

menu_col1, menu_col2, menu_col3, menu_col4, menu_col5 = st.columns(5)
with menu_col1:
    st.button(
        "Prehľad",
        key="menu_prehlad",
        use_container_width=True,
        type="primary" if current_menu_section == "Prehľad" else "secondary",
        on_click=set_top_menu_section,
        args=("Prehľad",)
    )
with menu_col2:
    st.button(
        "Výsledky modelov",
        key="menu_modely",
        use_container_width=True,
        type="primary" if current_menu_section == "Výsledky modelov" else "secondary",
        on_click=set_top_menu_section,
        args=("Výsledky modelov",)
    )
with menu_col3:
    st.button(
        "Asociačné pravidlá",
        key="menu_pravidla",
        use_container_width=True,
        type="primary" if current_menu_section == "Asociačné pravidlá" else "secondary",
        on_click=set_top_menu_section,
        args=("Asociačné pravidlá",)
    )
with menu_col4:
    st.button(
        "Prevalencia v populácii",
        key="menu_prevalencia",
        use_container_width=True,
        type="primary" if current_menu_section == "Prevalencia v populácii" else "secondary",
        on_click=set_top_menu_section,
        args=("Prevalencia v populácii",)
    )
with menu_col5:
    st.button(
        "Príručka",
        key="menu_prirucka",
        use_container_width=True,
        type="primary" if current_menu_section == "Príručka" else "secondary",
        on_click=set_top_menu_section,
        args=("Príručka",)
    )

menu_section = st.session_state["top_menu_section"]

# -----------------------------------------------
# NAČÍTANIE DÁT
# -----------------------------------------------
@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "Upraveny_dataset.xlsx"
    df = pd.read_excel(data_path)
    return df


@st.cache_data
def load_association_rules(file_path, file_mtime):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()


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
    # Plotly v niektorých kombináciách šablóny/trace vie zobraziť "undefined"
    # pre prázdne textové polia, preto ich explicitne normalizujeme.
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor=PALETTE['card'],
        plot_bgcolor=PALETTE['card'],
        font=dict(color=PALETTE['text']),
        title_font=dict(color=PALETTE['text']),
        legend=dict(
            font=dict(color=PALETTE['text']),
            title=dict(text=''),
            tracegroupgap=0
        ),
        legend_title_font=dict(color=PALETTE['text']),
        hoverlabel=dict(
            font=dict(color=PALETTE['text']),
            bgcolor=PALETTE['card'],
            bordercolor="#EBDCDC"
        ),
        title=dict(text=fig.layout.title.text or '')
    )
    fig.update_xaxes(
        gridcolor='#DCEBE9',
        zerolinecolor='#DCEBE9',
        tickfont=dict(color=PALETTE['text']),
        title_text=(fig.layout.xaxis.title.text or '')
    )
    fig.update_yaxes(
        gridcolor='#DCEBE9',
        zerolinecolor='#DCEBE9',
        tickfont=dict(color=PALETTE['text']),
        title_text=(fig.layout.yaxis.title.text or '')
    )
    fig.update_traces(
        textfont=dict(color=PALETTE['text'])
    )


def parse_itemset_text(value):
    if pd.isna(value):
        return []
    return re.findall(r"'([^']+)'", str(value))


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


@st.cache_data
def compute_rf_baseline_results(source_df):
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

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)

    labels = [1, 2, 3]
    rf_cm = confusion_matrix(y_test, y_pred_rf, labels=labels)
    dummy_cm = confusion_matrix(y_test, y_pred_dummy, labels=labels)

    return {
        "rf_acc": accuracy_score(y_test, y_pred_rf),
        "dummy_acc": accuracy_score(y_test, y_pred_dummy),
        "rf_f1_macro": f1_score(y_test, y_pred_rf, average='macro'),
        "dummy_f1_macro": f1_score(y_test, y_pred_dummy, average='macro', zero_division=0),
        "rf_cm": rf_cm,
        "dummy_cm": dummy_cm,
        "labels": labels
    }


def set_checkbox_group(keys, value):
    for key in keys:
        st.session_state[key] = value

# -----------------------------------------------
# SIDEBAR FILTRE
# -----------------------------------------------
st.sidebar.header("Filtre")

vysledok_col = 'Závažnosť priebehu ochorenia'

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
vysledok_options = sorted(df[vysledok_col].dropna().unique().tolist())
vysledok_labels = {
    1: '1 - Prepustenie domov',
    2: '2 - Presun na oddelenie',
    3: '3 - Smrť'
}
st.sidebar.markdown("**Výsledok hospitalizácie**")
vysledok_btn_col1, vysledok_btn_col2 = st.sidebar.columns(2)
vysledok_keys = [vysledok_labels.get(vysledok, str(vysledok)) for vysledok in vysledok_options]
with vysledok_btn_col1:
    st.button("Vybrať všetky", key="vysledok_select_all", use_container_width=True, on_click=set_checkbox_group, args=(vysledok_keys, True))
with vysledok_btn_col2:
    st.button("Zrušiť všetky", key="vysledok_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(vysledok_keys, False))
vysledok_selected = {}
for vysledok in vysledok_options:
    label = vysledok_labels.get(vysledok, str(vysledok))
    if label not in st.session_state:
        st.session_state[label] = True
    vysledok_selected[vysledok] = st.sidebar.checkbox(label, key=label)
vysledok_vyber = [value for value, selected in vysledok_selected.items() if selected]

# Komorbidity
st.sidebar.markdown("**Komorbidity**")
comorbidity_options = [
    'Hypertenzia',
    'Diabetes mellitus',
    'Kardiovaskulárne ochorenia',
    'Chronické respiračné ochorenia',
    'Renálne ochorenia',
    'Pečeňové ochorenia',
    'Onkologické ochorenia'
]
comorbidity_logic = st.sidebar.radio(
    "Logika komorbidít",
    options=["AND", "OR"],
    index=0,
    help="AND = pacient musí mať všetky vybrané komorbidity, OR = stačí aspoň jedna vybraná komorbidita."
)

comorbidity_btn_col1, comorbidity_btn_col2 = st.sidebar.columns(2)
comorbidity_keys = [f"comorbidity_{idx}" for idx, _ in enumerate(comorbidity_options)]
with comorbidity_btn_col1:
    st.button("Vybrať všetky", key="comorbidity_select_all", use_container_width=True, on_click=set_checkbox_group, args=(comorbidity_keys, True))
with comorbidity_btn_col2:
    st.button("Zrušiť všetky", key="comorbidity_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(comorbidity_keys, False))

comorbidity_selected = {}
for idx, comorbidity in enumerate(comorbidity_options):
    key = f"comorbidity_{idx}"
    if key not in st.session_state:
        st.session_state[key] = False
    comorbidity_selected[comorbidity] = st.sidebar.checkbox(comorbidity, key=key)

# Lieky
st.sidebar.markdown("**Lieky**")
drug_options = [
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
drug_logic = st.sidebar.radio(
    "Logika liekov",
    options=["AND", "OR"],
    index=0,
    help="AND = pacient musí mať všetky vybrané lieky, OR = stačí aspoň jeden vybraný liek."
)

drug_btn_col1, drug_btn_col2 = st.sidebar.columns(2)
drug_keys = [f"drug_{idx}" for idx, _ in enumerate(drug_options)]
with drug_btn_col1:
    st.button("Vybrať všetky", key="drug_select_all", use_container_width=True, on_click=set_checkbox_group, args=(drug_keys, True))
with drug_btn_col2:
    st.button("Zrušiť všetky", key="drug_deselect_all", use_container_width=True, on_click=set_checkbox_group, args=(drug_keys, False))

drug_selected = {}
for idx, drug_name in enumerate(drug_options):
    key = f"drug_{idx}"
    if key not in st.session_state:
        st.session_state[key] = False
    drug_selected[drug_name] = st.sidebar.checkbox(drug_name, key=key)

# -----------------------------------------------
# FILTROVANIE DÁT
# -----------------------------------------------
df_filtered = df[
    (df['Vek_kat'].isin(vek_vyber)) &
    (df['Pohlavie'].isin(pohlavie_vyber)) &
    (df['vlna'].isin(vlna_vyber)) &
    (df[vysledok_col].isin(vysledok_vyber))
]

# Komorbidity filter
selected_comorbidities = [
    comorbidity for comorbidity, selected in comorbidity_selected.items() if selected
]
valid_comorbidities = [
    comorbidity for comorbidity in selected_comorbidities if comorbidity in df_filtered.columns
]
if valid_comorbidities:
    if comorbidity_logic == "AND":
        mask = df_filtered[valid_comorbidities].eq(True).all(axis=1)
    else:
        mask = df_filtered[valid_comorbidities].eq(True).any(axis=1)
    df_filtered = df_filtered[mask]

# Lieky filter
selected_drugs = [drug for drug, selected in drug_selected.items() if selected]
valid_drugs = [drug for drug in selected_drugs if drug in df_filtered.columns]
if valid_drugs:
    if drug_logic == "AND":
        drug_mask = df_filtered[valid_drugs].eq(True).all(axis=1)
    else:
        drug_mask = df_filtered[valid_drugs].eq(True).any(axis=1)
    df_filtered = df_filtered[drug_mask]

# -----------------------------------------------
# CARDS - POČTY PACIENTOV
# -----------------------------------------------
if menu_section == "Prehľad":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Vybraní pacienti", len(df_filtered))
    with col2:
        st.metric("Celkovo pacientov", len(df))
    with col3:
        if len(df) > 0:
            st.metric("Podiel", f"{round(len(df_filtered)/len(df)*100, 1)} %")

    st.markdown("---")

    # -----------------------------------------------
    # GRAFY
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
        df_vysledok.loc[:, 'Výsledok_hospitalizácie_popis'] = df_vysledok[vysledok_col].map(zavaznost_map)
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

    fig_vlna_distribucia = px.bar(
        df_vlna_distribucia,
        x='vlna_label',
        y='Počet',
        text='Text',
        category_orders={'vlna_label': [str(v) for v in vlna_options]},
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

elif menu_section == "Výsledky modelov":
    st.subheader("Výsledky modelov: Random Forest vs Baseline")

    model_results = compute_rf_baseline_results(df)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("RF accuracy", f"{model_results['rf_acc'] * 100:.2f} %")
    with mcol2:
        st.metric("Baseline accuracy", f"{model_results['dummy_acc'] * 100:.2f} %")
    with mcol3:
        st.metric("RF macro F1", f"{model_results['rf_f1_macro']:.3f}")
    with mcol4:
        st.metric("Baseline macro F1", f"{model_results['dummy_f1_macro']:.3f}")

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**RF Confusion Matrix**")
        rf_cm_fig = px.imshow(
            model_results['rf_cm'],
            labels=dict(x="Predikovaná trieda", y="Skutočná trieda", color="Počet"),
            x=model_results['labels'],
            y=model_results['labels'],
            text_auto=True,
            color_continuous_scale=['#F6FBFF', '#E6F4FA', '#CBE8F3', '#A8D7EB', '#84C5E2']
        )
        apply_chart_theme(rf_cm_fig)
        rf_cm_fig.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color=PALETTE['text']),
                title=dict(text='Počet', font=dict(color=PALETTE['text']))
            )
        )
        rf_cm_fig.update_traces(textfont=dict(color=PALETTE['text']))
        st.plotly_chart(rf_cm_fig, use_container_width=True)

    with cm_col2:
        st.markdown("**Baseline Confusion Matrix**")
        dummy_cm_fig = px.imshow(
            model_results['dummy_cm'],
            labels=dict(x="Predikovaná trieda", y="Skutočná trieda", color="Počet"),
            x=model_results['labels'],
            y=model_results['labels'],
            text_auto=True,
            color_continuous_scale=['#F6FBFF', '#E6F4FA', '#CBE8F3', '#A8D7EB', '#84C5E2']
        )
        apply_chart_theme(dummy_cm_fig)
        dummy_cm_fig.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color=PALETTE['text']),
                title=dict(text='Počet', font=dict(color=PALETTE['text']))
            )
        )
        dummy_cm_fig.update_traces(textfont=dict(color=PALETTE['text']))
        st.plotly_chart(dummy_cm_fig, use_container_width=True)

elif menu_section == "Asociačné pravidlá":
    st.subheader("Asociačné pravidlá")
    rules_base_dir = Path(__file__).resolve().parent

    render_association_rules_section(
        "Apriori",
        str(rules_base_dir / "asociacne_pravidla.xlsx"),
        "apriori"
    )

    st.markdown("---")

    render_association_rules_section(
        "FP-Growth",
        str(rules_base_dir / "asociacne_pravidla_fpgrowth.xlsx"),
        "fpgrowth"
    )

elif menu_section == "Prevalencia v populácii":
    st.subheader("Prevalencia v populácii")

    if len(df) == 0:
        st.warning("Nie sú dostupné dáta na výpočet prevalencie.")
    else:
        prevalence_group = st.radio(
            "Zobraziť premenné",
            options=["Komorbidity", "Lieky", "Všetko"],
            horizontal=True,
            index=2
        )

        if prevalence_group == "Komorbidity":
            prevalence_cols = [col for col in comorbidity_options if col in df.columns]
        elif prevalence_group == "Lieky":
            prevalence_cols = [col for col in drug_options if col in df.columns]
        else:
            prevalence_cols = [
                col for col in (comorbidity_options + drug_options)
                if col in df.columns
            ]

        if not prevalence_cols:
            st.warning("V datasete sa nenašli požadované stĺpce pre prevalenciu.")
        else:
            total_population = len(df)
            filtered_population = len(df_filtered)

            total_positive = df[prevalence_cols].eq(True).sum()
            filtered_positive = df_filtered[prevalence_cols].eq(True).sum()

            total_prevalence = (total_positive / total_population * 100).fillna(0)
            if filtered_population > 0:
                filtered_prevalence = (filtered_positive / filtered_population * 100).fillna(0)
            else:
                filtered_prevalence = pd.Series(0.0, index=prevalence_cols)

            prevalence_table = pd.DataFrame({
                "Premenná": prevalence_cols,
                "Prevalencia celá populácia (%)": total_prevalence.reindex(prevalence_cols).round(2).values,
                "Prevalencia filtrovaná (%)": filtered_prevalence.reindex(prevalence_cols).round(2).values,
                "Rozdiel (p. b.)": (
                    filtered_prevalence.reindex(prevalence_cols) - total_prevalence.reindex(prevalence_cols)
                ).round(2).values,
                "Počet (celá)": total_positive.reindex(prevalence_cols).astype(int).values,
                "Počet (filtrovaná)": filtered_positive.reindex(prevalence_cols).astype(int).values
            })

            prevalence_table = prevalence_table.sort_values(
                "Prevalencia filtrovaná (%)",
                ascending=False
            )

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.metric("Veľkosť celej populácie", total_population)
            with pcol2:
                st.metric("Veľkosť filtrovanej populácie", filtered_population)
            with pcol3:
                if total_population > 0:
                    st.metric("Podiel filtrovanej populácie", f"{(filtered_population / total_population) * 100:.1f} %")

            st.dataframe(prevalence_table, use_container_width=True, hide_index=True)

            top_n = st.slider(
                "Počet premenných v grafe",
                min_value=5,
                max_value=min(20, len(prevalence_table)),
                value=min(10, len(prevalence_table)),
                step=1
            )

            chart_df = prevalence_table.head(top_n)
            prevalence_fig = px.bar(
                chart_df,
                x="Prevalencia filtrovaná (%)",
                y="Premenná",
                orientation="h",
                color="Rozdiel (p. b.)",
                text="Prevalencia filtrovaná (%)",
                color_continuous_scale=["#E7F3F1", "#9FD2CB", "#0F766E"]
            )
            prevalence_fig.update_traces(
                texttemplate="%{text:.2f} %",
                textposition="outside",
                cliponaxis=False
            )
            prevalence_fig.update_layout(
                margin=dict(l=24, r=24, t=24, b=64),
                coloraxis_colorbar=dict(title="Rozdiel (p. b.)")
            )
            prevalence_fig.update_yaxes(autorange="reversed", title_text="")
            apply_chart_theme(prevalence_fig)
            st.plotly_chart(prevalence_fig, use_container_width=True)

elif menu_section == "Príručka":
    st.subheader("Príručka")

    st.markdown(
        """
### Ako používať dashboard

1. Vľavo nastavte filtre (vek, pohlavie, vlna, výsledok hospitalizácie, komorbidity, lieky).
2. Hore si zvoľte sekciu podľa toho, čo chcete analyzovať.
3. Pri každej zmene filtrov sa grafy a tabuľky prepočítajú automaticky.

### Popis sekcií

- **Prehľad**: základná deskriptívna analýza filtrovanej kohorty.
- **Výsledky modelov**: porovnanie Random Forest modelu s baseline modelom.
- **Asociačné pravidlá**: pravidlá z Apriori a FP-Growth s nastaviteľnými prahmi.
- **Prevalencia v populácii**: porovnanie prevalencie komorbidít a liekov v celej vs filtrovanej populácii.

### Tipy k filtrom

- Pri komorbiditách a liekoch môžete voliť logiku **AND** alebo **OR**.
- Tlačidlá **Vybrať všetky** a **Zrušiť všetky** zrýchľujú prácu so sidebarom.
- Ak je filtrovaná populácia prázdna, upravte filtre na menej prísne.
        """
    )