# Systémová príručka

## Prehľad projektu

Táto systémová príručka popisuje technické požiadavky, štruktúru projektu a postup spustenia analytického riešenia vytvoreného ako súčasť bakalárskej práce.

Zdrojové klinické dáta nie sú súčasťou repozitára z dôvodu ochrany osobných údajov pacientov a musia byť vyžiadané samostatne.

Projekt spracováva anonymizované nemocničné dáta pacientov hospitalizovaných s ochorením COVID-19 počas štyroch vĺn pandémie. Zahŕňa prípravu a úpravu datasetu, exploratívnu analýzu, trénovanie klasifikačných modelov, hľadanie asociačných pravidiel a interaktívnu vizualizáciu výsledkov prostredníctvom Streamlit aplikácie.

---

## Systémové požiadavky

| Požiadavka | Hodnota |
|---|---|
| Operačný systém | Windows, Linux alebo macOS |
| Python | 3.10 alebo novší |
| Vývojové prostredie | JupyterLab 4.0+ |
| Minimálna RAM | 8 GB (odporúčané 16+ GB) |
| Voľné miesto na disku | približne 3 GB (vrátane virtuálneho prostredia) |

---

## Inštalácia a spustenie

Klonujte repozitár:

```bash
git clone https://github.com/kkuichi/mn513qk.git
cd mn513qk
```

Vytvorte virtuálne prostredie:

```bash
python -m venv .venv
```

Aktivujte virtuálne prostredie:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\activate
```

Nainštalujte závislosti:

```bash
pip install -r requirements.txt
```

Spustite JupyterLab:

```bash
jupyter lab
```

Po spustení sa v prehliadači otvorí interaktívne rozhranie JupyterLab, kde môžete otvárať a spúšťať jednotlivé notebooky z priečinka `Notebooks/`.

---

## 1. Úprava datasetu (`Upraveny dataset.ipynb`)

**Umiestnenie v repozitári:** [Notebooks/Upraveny dataset.ipynb](https://github.com/kkuichi/mn513qk/blob/main/Notebooks/Upraveny%20dataset.ipynb)  
**Výstup:** `Notebooks/Upraveny_dataset.xlsx`

### 1.1 Závislosti

| Knižnica | Účel |
|---|---|
| `pandas` | Načítanie, transformácia a export dát |
| `matplotlib` | Vizualizácia rozdelenia dát |

### 1.2 Vstupné súbory

Štyri Excel súbory reprezentujúce jednotlivé vlny pandémie COVID-19:

| Súbor | Vlna |
|---|---|
| `1Vlna.xlsx` | 1. vlna |
| `2Vlna.xlsx` | 2. vlna |
| `3Vlna.xlsx` | 3. vlna |
| `4Vlna.xlsx` | 4. vlna |

Súbory nie sú súčasťou repozitára z dôvodu ochrany osobných údajov a musia byť vyžiadané samostatne. Po získaní ich umiestnite do priečinka `Notebooks/`.

### 1.3 Postup spracovania

#### Krok 1 — Načítanie a zlúčenie dát

Každý súbor sa načíta do samostatného DataFrame, doplní sa stĺpec `vlna` (hodnoty 1–4) a všetky štyri DataFrame sa zlúčia do jedného pomocou `pd.concat`.

#### Krok 2 — Čistenie dát

Riadky s chýbajúcou hodnotou v stĺpci `Závažnosť priebehu ochorenia` (výsledok hospitalizácie) sú odstránené pomocou `dropna`.

#### Krok 3 — Výber stĺpcov

Z celého datasetu sa vyberú len relevantné stĺpce:

- **Demografické údaje:** `Pohlavie`, `Vek`, `vlna`
- **Výsledok hospitalizácie:** `Závažnosť priebehu ochorenia`
- **Komorbidity:** `Hypertenzia`, `Diabetes mellitus`, `Kardiovaskulárne ochorenia`, `Chronické respiračné ochorenia`, `Renálne ochorenia`, `Pečeňové ochorenia`, `Onkologické ochorenia`
- **Lieky:** MD652 \| FABIFLU TABLETS, MD656 IV-BECT 6MG (ivermectin), 5042D \| VEKLURY, 9547D \| PAXLOVID, LAGEVRIO, 00584 \| PYRIDOXIN LÉČIVA INJ, 24836 \| ACIDUM ASCORBICUM BBP, 89145 \| VITAMIN C-INJEKTOPAS, 24814 \| CALCIFEROL BBP 7,5 MG/ML, 92973 ALPHA D3, 00498 \| MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK, 00449 \| EREVIT 300 MG/ML, 02963 \| PREDNISON 20 LÉČIVA, 00269 \| PREDNISON 5 LÉČIVA, 84090 \| DEXAMED 6, 1275C \| DEXAMETAZÓN KRKA, MD661 BIODEXONE-DEXAMETHASONE, 2410B HYDROCORTISONE, 3242C \| OLUMIANT 4 MG, Anakinra, RoActemra, 34045 \| POLYOXIDONIUM 6 MG, 87299 \| IMUNOR, 56930 IMMODIN, Isoprinosine, 3879d INOMED, 35715 Azithromycin, 45954 Ceftriaxon, 0471B MOLOXIN, 9819A MOXIFLOXACIN, 58730 CIPROFLOXACIN KABI 200, 58746 CIPROFLOXACINKABI 400, 05044 OZZION, 4147C OMEMYL, 89662 NOLPAZA, 39397 PANTOPRAZOL, 94918 AMBROBENE, 24859 PENTOXYPHILLINUM, 8893 ACC INJEKT, 24949 CODEIN, 26846 OXANTIL, FRAXIPARIN, CLEXANE, FRAGMIN, ASPIRIN, ANOPYRIN

Prítomnosť komorbidity aj podanie lieku sú vyjadrené binárne hodnotami `True`/`False`.

#### Krok 4 — Zoskupenie liekov podľa pokynov lekárov

Lieky dostupné pod viacerými obchodnými názvami alebo v rôznych dávkach sa zlúčia do jedného stĺpca pomocou logického OR (`|`). Pôvodné stĺpce sa potom odstránia.

| Nový stĺpec | Zlúčené pôvodné stĺpce |
|---|---|
| `Vitamin C` | ACIDUM ASCORBICUM BBP, VITAMIN C-INJEKTOPAS |
| `Vitamin D` | CALCIFEROL BBP, ALPHA D3 |
| `Prednison` | PREDNISON 20 LÉČIVA, PREDNISON 5 LÉČIVA |
| `Dexametazon` | DEXAMED 6, DEXAMETAZÓN KRKA, BIODEXONE-DEXAMETHASONE |
| `Kineret` | Anakinra (detekcia z textového stĺpca `Liečba` pomocou `str.contains`) |
| `Isoprinosine/INOMED` | Isoprinosine, INOMED |
| `Moxifloxacin` | MOLOXIN, MOXIFLOXACIN |
| `Ciprofloxacin` | CIPROFLOXACIN KABI 200, CIPROFLOXACIN KABI 400 |
| `PPI` | OZZION, OMEMYL, NOLPAZA, PANTOPRAZOL |
| `Antikoagulancia` | FRAXIPARIN, CLEXANE, FRAGMIN |
| `Antiagregacne` | ASPIRIN, ANOPYRIN |

Po zlúčení liekov sa stĺpec `Liečba` (textový) odstráni.

#### Krok 5 — Kategorizácia veku

Stĺpec `Vek` (numerický) sa nahradí kategorickým stĺpcom `Vek_kat` podľa nasledovných intervalov:

| Kategória | Vekový rozsah |
|---|---|
| `18-44` | 18 – 44 rokov |
| `45-54` | 45 – 54 rokov |
| `55-64` | 55 – 64 rokov |
| `65-74` | 65 – 74 rokov |
| `75-84` | 75 – 84 rokov |
| `85+` | 85 a viac rokov |

Intervaly sú ľavostranné uzavreté (`right=False`). Pomocné stĺpce `vek_kategoria` a `vek_q6` sa odstránia.

#### Krok 6 — Export výsledného datasetu

Upravený dataset sa exportuje do súboru `Notebooks/Upraveny_dataset.xlsx` (bez indexového stĺpca).

### 1.4 Schéma výsledného datasetu (`Upraveny_dataset.xlsx`)

Dataset obsahuje 43 stĺpcov:

```
Pohlavie, Vek_kat, vlna, Závažnosť priebehu ochorenia,
Hypertenzia, Diabetes mellitus, Kardiovaskulárne ochorenia,
Chronické respiračné ochorenia, Renálne ochorenia,
Pečeňové ochorenia, Onkologické ochorenia,
MD652 | FABIFLU TABLETS, MD656 IV-BECT 6MG (ivermectin),
5042D | VEKLURY, 9547D | PAXLOVID, LAGEVRIO,
00584 | PYRIDOXIN LÉČIVA INJ, Vitamin C, Vitamin D,
00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK,
00449 | EREVIT 300 MG/ML, Prednison, Dexametazon,
2410B HYDROCORTISONE, 3242C | OLUMIANT 4 MG,
Kineret, RoActemra, 34045 | POLYOXIDONIUM 6 MG,
87299 | IMUNOR, 56930 IMMODIN, Isoprinosine/INOMED,
35715 Azithromycin, 45954 Ceftriaxon, Moxifloxacin,
Ciprofloxacin, PPI, 94918 AMBROBENE,
24859 PENTOXYPHILLINUM, 8893 ACC INJEKT,
24949 CODEIN, 26846 OXANTIL,
Antikoagulancia, Antiagregacne
```

---

## 2. Exploratívna analýza

### 2.1 Početnosti a chi-kvadrát test (`Pocetnosti_chi_kvadrat_test.ipynb`)

**Umiestnenie v repozitári:** [Notebooks/Pocetnosti_chi_kvadrat_test.ipynb](https://github.com/kkuichi/mn513qk/blob/main/Notebooks/Pocetnosti_chi_kvadrat_test.ipynb)  
**Výstup:** grafické vizualizácie (nie sú ukladané do súboru)

#### 2.1.1 Závislosti

| Knižnica | Účel |
|---|---|
| `pandas` | Načítanie a spracovanie dát |
| `matplotlib` | Vizualizácia grafov |
| `scipy.stats` | Chi-kvadrát test nezávislosti |
| `scipy.stats.contingency` | Výpočet Cramérovho V (sila asociácie) |

#### 2.1.2 Vstupné súbory

| Súbor | Vlna |
|---|---|
| `1Vlna.xlsx` | 1. vlna |
| `2Vlna.xlsx` | 2. vlna |
| `3Vlna.xlsx` | 3. vlna |
| `4Vlna.xlsx` | 4. vlna |

Súbory nie sú súčasťou repozitára z dôvodu ochrany osobných údajov a musia byť vyžiadané samostatne. Po získaní ich umiestnite do priečinka `Notebooks/`.

#### 2.1.3 Postup spracovania

#### Krok 1 — Načítanie a zlúčenie dát

Každý súbor sa načíta do samostatného DataFrame, doplní sa stĺpec `vlna` (hodnoty 1–4) a všetky štyri DataFrame sa zlúčia do jedného pomocou `pd.concat`.

#### Krok 2 — Čistenie dát

Riadky s chýbajúcou hodnotou v stĺpci `Závažnosť priebehu ochorenia` (výsledok hospitalizácie) sú odstránené pomocou `dropna`.

#### Krok 3 — Analýza pre každý liek a skupinu liekov

Pre každý liek aj skupinu liekov notebook vykoná nasledovný postup:

1. Pacienti sa rozdelia do dvoch skupín: **Dostali** / **Nedostali** daný liek
2. Vygenerujú sa **3 koláčové grafy** vedľa seba:
   - Rozdelenie výsledku hospitalizácie pre celý súbor
   - Rozdelenie výsledku hospitalizácie pre pacientov, ktorí liek dostali
   - Rozdelenie výsledku hospitalizácie pre pacientov, ktorí liek nedostali
3. Vykoná sa **chi-kvadrát test nezávislosti** medzi podaním lieku a výsledkom hospitalizácie
4. Vypočíta sa **Cramérovo V** ako miera sily asociácie

#### 2.1.4 Výstupy

Pre každý liek sa zobrazí trojdielny koláčový graf a výsledky štatistického testu (p-hodnota, Cramérovo V). Grafy nie sú automaticky ukladané do súborov.

---

### 2.2 Zmeny liečby počas vĺn (`Zmeny_vlny.ipynb`)

**Umiestnenie v repozitári:** [Notebooks/Zmeny_vlny.ipynb](https://github.com/kkuichi/mn513qk/blob/main/Notebooks/Zmeny_vlny.ipynb)  
**Výstup:** grafické vizualizácie (nie sú ukladané do súboru)

#### 2.2.1 Závislosti

| Knižnica | Účel |
|---|---|
| `pandas` | Načítanie a spracovanie dát |
| `numpy` | Numerické výpočty |
| `matplotlib` | Základné vykresľovanie |
| `seaborn` | Skupinové stĺpcové grafy |

#### 2.2.2 Vstupné súbory

| Súbor | Vlna |
|---|---|
| `1Vlna.xlsx` | 1. vlna |
| `2Vlna.xlsx` | 2. vlna |
| `3Vlna.xlsx` | 3. vlna |
| `4Vlna.xlsx` | 4. vlna |

Súbory nie sú súčasťou repozitára z dôvodu ochrany osobných údajov a musia byť vyžiadané samostatne. Po získaní ich umiestnite do priečinka `Notebooks/`.

#### 2.2.3 Postup spracovania

Pre každú skupinu liekov notebook:

1. Vypočíta percentuálne zastúpenie podania každého lieku v rámci každej vlny (`groupby('vlna').mean() * 100`)
2. Transformuje dáta do dlhého formátu pomocou `melt`
3. Vykreslí **skupinový stĺpcový graf** (seaborn `barplot`) s osou X = liek, farebným rozlíšením podľa vlny

#### 2.2.4 Analyzované lieky

Lieky sú analyzované po skupinách: Fabiflu, Ivermectin, Veklury, Paxlovid, Lagevrio, Pyridoxín, Vitamín C, Vitamín D, Magnézium, Erevit, Prednison, Dexametazón, Hydrokortizón, Olumiant, Anakinra, RoActemra, Polyoxidonium, Imunor, Immodin, Isoprinosine, INOMED, Azithromycin, Ceftriaxon, Moxifloxacin, Ciprofloxacin, PPI, Smecta, Reasec, Lagosa, Degan, Ambrobene, Pentoxyfylín, ACC, Kodeín, Oxantil, Fraxiparin, Clexane, Fragmin, Aspirin, Anopyrin.

#### 2.2.5 Výstupy

Pre každú skupinu sa zobrazí jeden skupinový stĺpcový graf znázorňujúci, ako sa percentuálne zastúpenie podávania liekov menilo naprieč 1. – 4. vlnou pandémie.

---

## 3. Streamlit aplikácia

**Umiestnenie v repozitári:** [https://github.com/kkuichi/mn513qk](https://github.com/kkuichi/mn513qk)

### 3.1 Štruktúra repozitára

```
mn513qk/
├── .devcontainer/
├── Guides/             # systémová a používateľská príručka — nie je súčasťou aplikácie
├── Notebooks/          # príprava dát a exploratívna analýza — nie je súčasťou aplikácie
├── pages/
│   ├── 1_Prehľad_pacientov.py
│   ├── 2_Výsledky_modelov.py
│   ├── 3_Asociačné_pravidlá.py
│   └── 4_Prevalencia_komorbidít.py
├── Upraveny_dataset.xlsx
├── asociacne_pravidla.xlsx
├── asociacne_pravidla_fpgrowth.xlsx
├── requirements.txt
├── utils.py
└── Úvod.py
```

| Súbor / priečinok | Popis |
|---|---|
| `Úvod.py` | Vstupný bod aplikácie — nastavuje konfiguráciu stránky, aplikuje vizuálny štýl a zobrazuje úvodný prehľad sekcií |
| `utils.py` | Centrálny modul zdieľaný naprieč stránkami — načítanie dát, trénovanie modelov, logika filtrov, vizuálny štýl a pomocné funkcie |
| `pages/` | Podstránky aplikácie |
| `Upraveny_dataset.xlsx` | Upravený dataset (vstup pre aplikáciu) |
| `asociacne_pravidla.xlsx` | Asociačné pravidlá — algoritmus Apriori |
| `asociacne_pravidla_fpgrowth.xlsx` | Asociačné pravidlá — algoritmus FP-Growth |
| `requirements.txt` | Zoznam závislostí |
| `.devcontainer/` | Konfigurácia vývojového kontajnera |

### 3.2 Závislosti

| Knižnica | Účel |
|---|---|
| `streamlit` | Webový framework aplikácie |
| `pandas` | Načítanie a spracovanie dát |
| `plotly` | Interaktívne grafy |
| `scikit-learn` | Klasifikačné modely a metriky |
| `xgboost` | XGBoost klasifikátor |
| `openpyxl` | Čítanie Excel súborov |

### 3.3 Vstupné dáta

Aplikácia pri spustení načítava tri súbory umiestnené v koreni repozitára:

| Súbor | Zdroj |
|---|---|
| `Upraveny_dataset.xlsx` | Výstup notebooku `Notebooks/Upraveny dataset.ipynb` |
| `asociacne_pravidla.xlsx` | Výstup notebooku `Notebooks/Asociacne_pravidla.ipynb` (algoritmus Apriori) |
| `asociacne_pravidla_fpgrowth.xlsx` | Výstup notebooku `Notebooks/Asociacne_pravidla.ipynb` (algoritmus FP-Growth) |

### 3.4 Spustenie aplikácie

```bash
streamlit run Úvod.py
```

Po spustení sa aplikácia otvorí v prehliadači na adrese `http://localhost:8501`.

### 3.5 Sekcie aplikácie

Aplikácia obsahuje štyri sekcie: **Prehľad pacientov**, **Výsledky modelov**, **Asociačné pravidlá** a **Prevalencia komorbidít**. Podrobný popis každej sekcie a návod na používanie je uvedený v Používateľskej príručke (`Guides/Používateľská príručka.md`).

---

## Poznámky k spusteniu

- Surové dátové súbory (`1Vlna.xlsx` – `4Vlna.xlsx`) nie sú súčasťou repozitára z dôvodu ochrany osobných údajov. Po vyžiadaní ich umiestnite do priečinka `Notebooks/`.
- Notebook `Notebooks/Upraveny dataset.ipynb` musí byť spustený ako prvý — jeho výstup `Notebooks/Upraveny_dataset.xlsx` je vstupom pre ďalšie notebooky.
- Notebooky exploratívnej analýzy (`Pocetnosti_chi_kvadrat_test.ipynb`, `Zmeny_vlny.ipynb`) načítavajú surové Excel súbory priamo a nevyžadujú `Upraveny_dataset.xlsx`.
- Spustenie všetkých buniek v každom notebooku prebieha sekvenčne — každý krok závisí na predchádzajúcom.
