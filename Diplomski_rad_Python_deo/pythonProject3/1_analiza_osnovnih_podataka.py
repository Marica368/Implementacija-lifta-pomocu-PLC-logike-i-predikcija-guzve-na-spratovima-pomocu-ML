import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Učitavanje podataka
df = pd.read_csv('dataset.csv')

# 2. Čišćenje (samo radni dani i radno vreme)
df_clean = df[df['dan_u_nedelji'] < 5].copy()
df_clean = df_clean[(df_clean['sat'] >= 6) & (df_clean['sat'] <= 19)]

# Kreiram vremenski slot od 30 minuta
# Ako je minut < 30, slot je "00", ako je >= 30, slot je "30"
df_clean['slot'] = df_clean['minut'].apply(lambda x: '00' if x < 30 else '30')
# Spajam sat i slot u jednu kolonu, npr. "10:00" ili "10:30"
df_clean['vreme_detaljno'] = df_clean['sat'].astype(str) + ":" + df_clean['slot']

# 3. Vizuelizacija
plt.figure(figsize=(16, 8))

# Sortiram vreme da ide hronološki
vremenske_etikete = []
for h in range(6, 20):
    vremenske_etikete.append(f"{h}:00")
    vremenske_etikete.append(f"{h}:30")

# Pravim pivot tabelu sa novim detaljnim vremenom
heatmap_data = df_clean.pivot_table(index='poziv_sa_sprata',
                                    columns='vreme_detaljno',
                                    aggfunc='size',
                                    fill_value=0)

# Ređam kolone po hronološkim etiketama
heatmap_data = heatmap_data.reindex(columns=vremenske_etikete)

sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='d')
plt.title('Detaljna toplotna mapa (Rezolucija 30 min): Doručak vs. Kraj doručka')
plt.xlabel('Vreme (Sat:Minut)')
plt.ylabel('Sprat')
plt.xticks(rotation=45)
plt.savefig('toplotna_mapa.png')
plt.show()