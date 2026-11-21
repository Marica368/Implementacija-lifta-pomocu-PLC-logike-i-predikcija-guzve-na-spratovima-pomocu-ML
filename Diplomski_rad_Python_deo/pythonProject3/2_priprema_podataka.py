"""
Ukratko:
  1. Ciklicno kodiranje vremena (sin/cos) - model "zna" da je 23:59
     blizu 00:00, umesto da tretira vreme kao pravu liniju od 0 do 1439.
  2. Dan u nedelji se tretira kao KATEGORIJA (one-hot), ne kao broj,
     jer petak (4) nije "veci" od ponedeljka (0).
  3. Peak-flegovi
"""

import pandas as pd
import numpy as np

# --- 1. Ucitavanje ---
df = pd.read_csv('dataset.csv')

# Samo radni dani
df = df[df['dan_u_nedelji'] < 5].copy()

# --- 2. Vreme u minutima od ponoci ---
df['ukupno_minuta'] = df['sat'] * 60 + df['minut']

# --- 3. Ciklicno kodiranje vremena (sin/cos) ---
minuta_u_danu = 24 * 60
df['sin_vreme'] = np.sin(2 * np.pi * df['ukupno_minuta'] / minuta_u_danu)
df['cos_vreme'] = np.cos(2 * np.pi * df['ukupno_minuta'] / minuta_u_danu)

# --- 4. Peak-flegovi ---
df['is_morning_peak'] = df['ukupno_minuta'].between(420, 480).astype(int)    # 07-08h
df['is_lunch_peak'] = df['ukupno_minuta'].between(600, 660).astype(int)      # 10-11h
df['is_afternoon_peak'] = df['ukupno_minuta'].between(900, 960).astype(int)  # 15-16h
df['is_radno_vreme'] = df['ukupno_minuta'].between(480, 1020).astype(int)    # 08-17h

# --- 5. Dan u nedelji kao kategorija (one-hot), ne kao broj ---
dan_dummies = pd.get_dummies(df['dan_u_nedelji'], prefix='dan').astype(int)
df = pd.concat([df, dan_dummies], axis=1)

# Kolonu 'dan_u_nedelji' izbacujem, odmah nakon sto sam je iskoristila da napravim dan_0...dan_4.
df = df.drop(columns=['dan_u_nedelji'])

# --- 6. Cuvanje rezultata za sledeci fajl ---
df.to_csv('podaci_pripremljeni.csv', index=False)

print("--- PODACI NAKON PRIPREME ---")
print(df.head(10))
print(f"\nBroj redova: {len(df)}")
print(f"Kolone u fajlu: {list(df.columns)}")
print("\nRaspodela klasa (sprat):")
print(df['poziv_sa_sprata'].value_counts().sort_index())
print("\n[Sacuvano] podaci_pripremljeni.csv")
