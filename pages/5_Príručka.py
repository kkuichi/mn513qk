import streamlit as st

from utils import setup_page

setup_page("Príručka")

st.title("Príručka")

st.markdown(
    """
### Ako používať dashboard

1. Vľavo nastavte filtre (vek, pohlavie, vlna, výsledok hospitalizácie, komorbidity, lieky).
2. Hore si zvoľte sekciu podľa toho, čo chcete analyzovať.
3. Pri každej zmene filtrov sa grafy a tabuľky prepočítajú automaticky.

### Popis sekcií

- **Interaktívne grafy**: základná deskriptívna analýza filtrovanej kohorty.
- **Výsledky modelov**: porovnanie Random Forest modelu s baseline modelom.
- **Asociačné pravidlá**: pravidlá z Apriori a FP-Growth s nastaviteľnými prahmi.
- **Prevalencia komorbidít**: porovnanie prevalencie komorbidít v celej vs filtrovanej populácii.

### Tipy k filtrom

- Pri komorbiditách a liekoch môžete voliť logiku **AND** alebo **OR**.
- Tlačidlá **Vybrať všetky** a **Zrušiť všetky** zrýchľujú prácu so sidebarom.
- Ak je filtrovaná populácia prázdna, upravte filtre na menej prísne.
    """
)
