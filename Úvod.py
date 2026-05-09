import streamlit as st

from utils import setup_page

setup_page("COVID-19 Dashboard")

st.title("Vitajte")

st.markdown(
    """
    Tento interaktívny dashboard slúži na vizualizáciu a analýzu dát o pacientoch hospitalizovaných s ochorením COVID-19 v Univerzitnej nemocnici Louisa Pasteura v Košiciach. 
    
    Dashboard pracuje s redukovaným datasetom 3 845 pacientov so **zameraním na podávané lieky**. Dataset obsahuje informácie o podávaných liekoch, komorbiditách, veku (rozdelenom do kategórií), pohlaví, pandemickej vlne a výsledku hospitalizácie.
    """
)

st.markdown(
    """
    ### Prehľad pacientov
    Sekcia poskytuje **deskriptívny pohľad** na zloženie kohorty pacientov prostredníctvom **interaktívnych grafov**. Pomocou filtrov v ľavom paneli je možné kohortu cielene zúžiť.

    ### Výsledky modelov
    Sekcia prezentuje **výkonnosť troch prediktívnych modelov** (Random Forest, XGBoost, Logistická regresia) v porovnaní s baseline modelom. Cieľom bolo zistiť, do akej miery je možné na základe podávaných liekov a ďalších dostupných údajov predikovať výsledok hospitalizácie.

    ### Asociačné pravidlá
    Sekcia zobrazuje pravidlá objavené algoritmami Apriori a FP-Growth, ktoré odhaľujú vzťahy medzi jednotlivými atribútmi v datasete. Slúžia na identifikáciu prípadných **vzorov** v podávaní liekov a ich vplyvu na priebeh ochorenia.

    ### Prevalencia komorbidít
    Sekcia porovnáva prevalenciu jednotlivých komorbidít v datasete s prevalenciou v bežnej populácii podľa odbornej literatúry. Slúži ako doplnkový kontextový pohľad na charakteristiku skúmanej kohorty.

    """
)
