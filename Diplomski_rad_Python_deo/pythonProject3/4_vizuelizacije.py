"""
VIZUELIZACIJE
  1_korelacija.png
  2_tacnost_modela.png
  3_vaznost_atributa.png
  4_matrica_konfuzije.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix


# 1. UCITAVANJE SVEGA STO JE PRETHODNI FAJL SACUVAO

rezime = pd.read_csv('rezime_modela.csv')
rezultati_testa = pd.read_csv('rezultati_testa.csv')
vaznost = pd.read_csv('znacajnost_atributa.csv')
X = pd.read_csv('atributi_X.csv')

rf_model = joblib.load('rf_model.pkl')
stablo_model = joblib.load('stablo_model.pkl')

# GRAFIKON 1: Korelaciona mapa SA ciljnom promenljivom

# Ucitavam kompletan dataset iz faze 2 (koji sadrzi i atribute i ciljnu promenljivu 'poziv_sa_sprata')
df_full = pd.read_csv('podaci_pripremljeni.csv')

# Crtanje mape
plt.figure(figsize=(12, 10))
korelacije_full = df_full.corr()

# Pravimo masku za gornji trougao da se ne dupliraju vrednosti
mask_full = np.triu(np.ones_like(korelacije_full, dtype=bool), k=1)

sns.heatmap(korelacije_full, mask=mask_full, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})

plt.title('Korelaciona mapa', fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('1_korelacija.png', dpi=150)
plt.close()
print("[Sacuvano] 1_korelacija.png")


# GRAFIKON 2: Poredjenje tacnosti modela

plt.figure(figsize=(9, 6))
imena = rezime['Model'].tolist()
cv_tacnosti = rezime['CV tacnost (%)'].tolist()
cv_std = rezime['CV std (%)'].tolist()
bars = plt.bar(imena, cv_tacnosti, yerr=cv_std, capsize=6,
                color=['#4C72B0', '#DD8452', '#55A868'])
plt.ylabel('5-fold CV tacnost (%)')
plt.title('2. Poredjenje preciznosti modela')
plt.ylim(0, max(cv_tacnosti) + 15)
for bar, val in zip(bars, cv_tacnosti):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.1f}%',
              ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('2_tacnost_modela.png', dpi=150)
plt.close()
print("[Sacuvano] 2_tacnost_modela.png")


# GRAFIKON 3: Znacaj atributa (Random Forest)

vaznost_sort = vaznost.sort_values('znacaj', ascending=True)
plt.figure(figsize=(9, 7))
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(vaznost_sort)))
plt.barh(vaznost_sort['atribut'], vaznost_sort['znacaj'], color=colors)
plt.title('3. Znacaj atributa (feature importance) - Random Forest')
plt.xlabel('Relativni znacaj')
for i, v in enumerate(vaznost_sort['znacaj']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('3_vaznost_atributa.png', dpi=150)
plt.close()
print("[Sacuvano] 3_vaznost_atributa.png")


# GRAFIKON 4: Matrica konfuzije

najbolje_ime = rezime.loc[rezime['CV tacnost (%)'].idxmax(), 'Model']
kolona_pred = f"predikcija_{najbolje_ime.replace(' ', '_')}"

plt.figure(figsize=(8, 6))
cm = confusion_matrix(rezultati_testa['stvarno'], rezultati_testa[kolona_pred])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.title(f'4. Matrica konfuzije (Najbolji model: {najbolje_ime})')
plt.xlabel('Predvidjeni sprat (AI)')
plt.ylabel('Stvarni sprat (Korisnik)')
plt.tight_layout()
plt.savefig('4_matrica_konfuzije.png', dpi=150)
plt.close()
print(f"[Sacuvano] 4_matrica_konfuzije.png (najbolji model: {najbolje_ime})")

print("\nSve vizuelizacije su gotove.")
