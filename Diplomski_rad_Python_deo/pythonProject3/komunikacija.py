"""
MAPA REGISTARA (iz CoDeSys):
  --- COILS (bool signali) ---
  Coil 0 (Coils[0].Bit0) -> bAI_Mode
  Coil 1 (Coils[0].Bit1) -> bAI_Request
  Coil 2 (Coils[0].Bit2) -> bAI_ResponseReady  <- OVDE PYTHON UPISUJE

  --- HOLDING REGISTERS (celi brojevi) ---
  Holding Register 0  -> iCurrentHour       (0-23)
  Holding Register 1  -> iCurrentMinute     (0-59)
  Holding Register 2  -> iCurrentDay        (0=Pon...6=Ned)
  Holding Register 3  -> iAI_PredictedFloor (0-3)  <- OVDE PYTHON UPISUJE

"""

import time
import joblib
import numpy as np
import pandas as pd
from pymodbus.client import ModbusTcpClient


# 1. PODESAVANJA KONEKCIJE

IP_ADRESA = "127.0.0.1"  # localhost, jer CoDeSys simulacija radi na istom racunaru
PORT = 502  # standardni Modbus TCP port

# --- COILS (bool signali) - Bit0, Bit1, Bit2 unutar Coils[0]
COIL_AI_MODE = 0
COIL_AI_REQUEST = 1
COIL_AI_RESPONSE_READY = 2

# --- HOLDING REGISTERS (celi brojevi)
REG_HOUR = 0
REG_MINUTE = 1
REG_DAY = 2
REG_PREDICTED_FLOOR = 3

# Koliko cesto Python proverava CoDeSys (u sekundama)
INTERVAL_PROVERE = 1.0


# 2. UCITAVANJE VEC ISTRENIRANOG MODELA

# Koristimo stablo_model.pkl koji je vec istreniran
print("Ucitavam istreniran model (stablo_model.pkl)...")
model = joblib.load('stablo_model.pkl')
print("Model ucitan.\n")


def napravi_atribute(sat, minut, dan):
    """
    Pretvara sirovo vreme (sat, minut, dan) u isti oblik atributa
    koji je model video tokom treninga (2_priprema_podataka.py).
    """
    ukupno_minuta = sat * 60 + minut
    minuta_u_danu = 24 * 60
    sin_vreme = np.sin(2 * np.pi * ukupno_minuta / minuta_u_danu)
    cos_vreme = np.cos(2 * np.pi * ukupno_minuta / minuta_u_danu)

    is_morning_peak = 1 if 420 <= ukupno_minuta < 480 else 0  # 07-08h
    is_lunch_peak = 1 if 600 <= ukupno_minuta < 660 else 0  # 10-11h
    is_afternoon_peak = 1 if 900 <= ukupno_minuta < 960 else 0  # 15-16h
    is_radno_vreme = 1 if 480 <= ukupno_minuta < 1020 else 0  # 08-17h

    # One-hot dani - mora imati isti broj kolona kao u treningu (dan_0..dan_4)
    dan_kolone = [1 if dan == i else 0 for i in range(5)]

    # VAZAN REDOSLED
    # ['ukupno_minuta', 'sin_vreme', 'cos_vreme', 'is_morning_peak',
    #  'is_lunch_peak', 'is_afternoon_peak', 'is_radno_vreme',
    #  'dan_0', 'dan_1', 'dan_2', 'dan_3', 'dan_4']
    imena_kolona = ['ukupno_minuta', 'sin_vreme', 'cos_vreme', 'is_morning_peak',
                    'is_lunch_peak', 'is_afternoon_peak', 'is_radno_vreme',
                    'dan_0', 'dan_1', 'dan_2', 'dan_3', 'dan_4']
    vrednosti = [ukupno_minuta, sin_vreme, cos_vreme, is_morning_peak,
                 is_lunch_peak, is_afternoon_peak, is_radno_vreme] + dan_kolone

    # Koristimo DataFrame (ne obicnu listu) da model dobije ista imena
    # kolona kao tokom treninga - ovo sprecava UserWarning upozorenje
    # i cini kod otporniji na buduce izmene redosleda atributa.
    return pd.DataFrame([vrednosti], columns=imena_kolona)


def procitaj_coil(client, adresa):
    # Cita jedan Coil (BOOL signal) i vraca True/False
    rezultat = client.read_coils(address=adresa, count=1)
    if rezultat.isError():
        raise ConnectionError(f"Greska pri citanju coil-a {adresa}")
    return rezultat.bits[0]


def upisi_coil(client, adresa, vrednost):
    # Upisuje bool vrednost (True/False) u jedan Coil
    rezultat = client.write_coil(address=adresa, value=bool(vrednost))
    if rezultat.isError():
        raise ConnectionError(f"Greska pri upisu u coil {adresa}")


def procitaj_registar(client, adresa):
    # Cita jedan Holding Register i vraca ga kao ceo broj
    rezultat = client.read_holding_registers(address=adresa, count=1)
    if rezultat.isError():
        raise ConnectionError(f"Greska pri citanju registra {adresa}")
    return rezultat.registers[0]


def upisi_registar(client, adresa, vrednost):
    # Upisuje celobrojnu vrednost u jedan Holding Register
    rezultat = client.write_register(address=adresa, value=int(vrednost))
    if rezultat.isError():
        raise ConnectionError(f"Greska pri upisu u registar {adresa}")



# 3. GLAVNA PETLJA - POVEZIVANJE I STALNO OSLUSKIVANJE

def main():
    client = ModbusTcpClient(IP_ADRESA, port=PORT)

    print(f"Povezujem se na CoDeSys ({IP_ADRESA}:{PORT})...")
    if not client.connect():
        print("GRESKA: Ne mogu da se povezem. Proveri da li je CoDeSys "
              "simulacija pokrenuta i da li Modbus TCP Slave radi.")
        return
    print("Povezano! Cekam AI_Request signal od CoDeSys-a...\n")

    try:
        while True:
            try:
                ai_mode = procitaj_coil(client, COIL_AI_MODE)
                ai_request = procitaj_coil(client, COIL_AI_REQUEST)

                if ai_mode and ai_request:
                    sat = procitaj_registar(client, REG_HOUR)
                    minut = procitaj_registar(client, REG_MINUTE)
                    dan = procitaj_registar(client, REG_DAY)

                    print(f"[AI zahtev] Vreme: {sat:02d}:{minut:02d}, dan={dan} "
                          f"-> racunam predikciju...")

                    X_novo = napravi_atribute(sat, minut, dan)
                    predikcija = int(model.predict(X_novo)[0])

                    upisi_registar(client, REG_PREDICTED_FLOOR, predikcija)
                    upisi_coil(client, COIL_AI_RESPONSE_READY, True)

                    print(f"[AI odgovor] Predvidjen sprat: {predikcija} "
                          f"-> poslato CoDeSys-u.\n")

                    # Cekam da CoDeSys obradi i resetuje AI_Request,
                    # da ne bismo slali istu predikciju u krug dok
                    # CoDeSys ne stigne da je iskoristi.
                    time.sleep(2.0)
                else:
                    # Nema AI zahteva - samo cekamo sledecu proveru
                    pass

            except ConnectionError as e:
                print(f"Greska u komunikaciji: {e}. Pokusavam ponovo...")
                time.sleep(2.0)

            time.sleep(INTERVAL_PROVERE)

    except KeyboardInterrupt:
        print("\nPrekidam vezu.")
    finally:
        client.close()
        print("Konekcija zatvorena.")


if __name__ == '__main__':
    main()