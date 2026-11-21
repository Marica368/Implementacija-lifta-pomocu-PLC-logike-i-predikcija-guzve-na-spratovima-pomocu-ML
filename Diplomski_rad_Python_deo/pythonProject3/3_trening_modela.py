"""
TRENING MODELA

Modeli koji se treniraju i porede:
  1. Stablo Odlucivanja
  2. Random Forest
  3. k-NN

"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# 1. UCITAVANJE PRIPREMLJENIH PODATAKA

df = pd.read_csv('podaci_pripremljeni.csv')

# Uzimam sve kolone koje pocinju sa "dan_" - to su tacno dan_0...dan_4
dan_kolone = [c for c in df.columns if c.startswith('dan_')]
kolone_X = ['ukupno_minuta', 'sin_vreme', 'cos_vreme',
            'is_morning_peak', 'is_lunch_peak', 'is_afternoon_peak',
            'is_radno_vreme'] + dan_kolone

X = df[kolone_X]
y = df['poziv_sa_sprata']

print(f"Broj atributa: {X.shape[1]}")
print(f"Atributi: {kolone_X}")


# 2. PODELA NA TRENING I TEST (stratifikacija)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STANDARDIZACIJA - potrebna za k-NN (osetljiv je na brojeve)
scaler = StandardScaler()
scaler.fit(X_train) # Računa srednju vrednost i standardnu devijaciju za svaku kolonu
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. TRENING MODELA

rezultati = []

def evaluiraj(naziv, model, Xtr, Xte, X_cv):
    # Treniranje modela
    model.fit(Xtr, y_train)
    # Predikcija nad test skupom
    y_pred = model.predict(Xte)
    # Računanje tačnosti
    tacnost = accuracy_score(y_test, y_pred)
    # 5-FOLD CROSS-VALIDATION
    cv_scores = cross_val_score(model, X_cv, y, cv=5, n_jobs=-1)

    print(f"\n{'=' * 50}")
    print(f"MODEL: {naziv}")
    print(f"{'=' * 50}")
    print(f"Tacnost na test skupu:      {tacnost * 100:.2f}%")
    print(f"5-fold CV prosečna tacnost : {cv_scores.mean() * 100:.2f}% "
          f"(+/- {cv_scores.std() * 100:.2f}%)")
    print("\nIzvestaj po klasama (sprat):")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        'naziv': naziv, 'model': model, 'y_pred': y_pred,
        'tacnost_test': tacnost, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
    }

# --- Stablo Odlucivanja  ---
stablo_model = DecisionTreeClassifier(max_depth=5, random_state=42)
rezultati.append(evaluiraj('Stablo Odlucivanja', stablo_model, X_train, X_test, X))

# --- Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rezultati.append(evaluiraj('Random Forest', rf_model, X_train, X_test, X))

# --- k-NN (radi na skaliranim podacima) ---
knn_model = KNeighborsClassifier(n_neighbors=5)
X_scaled_ceo = scaler.transform(X)
rezultati.append(evaluiraj('k-NN', knn_model, X_train_scaled, X_test_scaled, X_scaled_ceo))


# 4. NAJBOLJI MODEL

najbolji = max(rezultati, key=lambda r: r['cv_mean'])
print(f"\n{'#' * 50}")
print(f"NAJBOLJI MODEL: {najbolji['naziv']} ({najbolji['cv_mean'] * 100:.2f}% CV tacnost)")
print(f"{'#' * 50}")


# 5. CUVANJE SVEGA

joblib.dump(stablo_model, 'stablo_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Rezultati testa (stvarno vs predvidjeno) - za matrice konfuzije
rez_test = pd.DataFrame({'stvarno': y_test.values})
for r in rezultati:
    rez_test[f"predikcija_{r['naziv'].replace(' ', '_')}"] = r['y_pred']
rez_test.to_csv('rezultati_testa.csv', index=False)

# Rezime tacnosti
rezime = pd.DataFrame([{
    'Model': r['naziv'],
    'Test tacnost (%)': round(r['tacnost_test'] * 100, 2),
    'CV tacnost (%)': round(r['cv_mean'] * 100, 2),
    'CV std (%)': round(r['cv_std'] * 100, 2),
} for r in rezultati])
rezime.to_csv('rezime_modela.csv', index=False)
print("\n--- REZIME ---")
print(rezime.to_string(index=False))

# Znacaj atributa (iz Random Foresta)
vaznost = pd.DataFrame({
    'atribut': kolone_X,
    'znacaj': rf_model.feature_importances_
}).sort_values('znacaj', ascending=False)
vaznost.to_csv('znacajnost_atributa.csv', index=False)

# Cuvamo i X (sa nazivima kolona) da ga sledeci fajl koristi za korelacionu mapu
X.to_csv('atributi_X.csv', index=False)

print("\n[Sacuvano] stablo_model.pkl, rf_model.pkl, knn_model.pkl, scaler.pkl")
print("[Sacuvano] rezultati_testa.csv, rezime_modela.csv, znacajnost_atributa.csv, atributi_X.csv")

