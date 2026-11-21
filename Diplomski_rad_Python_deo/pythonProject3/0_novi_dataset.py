import pandas as pd
import numpy as np
import random

naziv_fajla = "dataset.csv"
data = []

# Simuliram 60 dana
for dan_u_mesecu in range(60):
    dan_u_nedelji = dan_u_mesecu % 7  # 0=Pon, 6=Ned

    for sat in range(24):
        for minut in range(60):

            # Definišem verovatnoću poziva (učestanost)
            verovatnoca_poziva = 0.01  # Osnovna (noćna/vikend) tišina

            if dan_u_nedelji < 5:  # Radni dani
                if 7 <= sat < 8:
                    verovatnoca_poziva = 0.90  # Jaka učestanost
                elif 10 <= sat < 11:
                    verovatnoca_poziva = 0.60  # Srednja učestanost
                elif 15 <= sat < 16:
                    verovatnoca_poziva = 0.80  # Srednjе jaka učestanost
                elif 8 <= sat < 17:
                    verovatnoca_poziva = 0.15  # Normalna radna učestanost

            # Ako je nasumičan broj manji od verovatnoće, desi se poziv
            if random.random() < verovatnoca_poziva:

                # Određivanje sprata
                if dan_u_nedelji < 5:
                    if 7 <= sat < 8:
                        poziv_sa_sprata = 0
                    elif sat == 10 and minut <= 30:
                        poziv_sa_sprata = random.choice([2, 3])
                    elif sat == 10 and minut > 30:
                        poziv_sa_sprata = 1
                    elif 15 <= sat < 16:
                        poziv_sa_sprata = random.choice([2, 3])
                    else:
                        # Čak i van špica, sprat 0 je najčešći (ulaz u zgradu)
                        poziv_sa_sprata = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.1, 0.3])
                else:
                    poziv_sa_sprata = random.randint(0, 3)

                data.append([dan_u_nedelji, sat, minut, poziv_sa_sprata])

df = pd.DataFrame(data, columns=['dan_u_nedelji', 'sat', 'minut', 'poziv_sa_sprata'])
df.to_csv(naziv_fajla, index=False)
print(f"Generisan je dataset sa {len(df)} poziva.")