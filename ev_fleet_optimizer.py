import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
import warnings
import seaborn as sns

# SIMULATION_START = datetime(2024, 8, 19, 0, 0)

# Ignorovanie zbytočných varovaní
warnings.filterwarnings('ignore')
FLEET_SIZE = 30  # Počet vozidiel vo flotile
WORKDAYS_ONLY = True  # Ak True, vozidlá sa pohybujú len počas pracovných dní (pondelok-piatok), nie počas víkendov
SIMULATION_START = datetime(2024, 8, 19, 0, 0)
SIMULATION_DAYS = 7 # Počet dní na simuláciu
DAM_ALLOCATION=0.05 # Percento energie alokovanej na DAM

class EVFleetOptimizer:
    """
    Optimalizátor nabíjania flotily elektrických vozidiel s využitím cenových rozdielov
    medzi denným trhom (DAM) a vnútrodenným trhom (IDM).
    """

    def __init__(self, fleet_size=10, ev_capacity=47, min_charge_kw=1.5, max_charge_kw=11, min_soc_target=0.9,
                 workdays_only=False):
        """
        Inicializácia optimalizátora s parametrami flotily

        Parametre:
        -----------
        fleet_size : int
            Počet vozidiel vo flotile
        ev_capacity : float
            Kapacita batérie každého vozidla v kWh
        min_charge_kw : float
            Minimálny nabíjací výkon na vozidlo v kW
        max_charge_kw : float
            Maximálny nabíjací výkon na vozidlo v kW
        min_soc_target : float
            Minimálny cieľový stav nabitia (0-1), ktorý musí byť dosiahnutý pred odchodom
        workdays_only : bool
            Ak True, vozidlá sa pohybujú len počas pracovných dní (pondelok-piatok), nie počas víkendov
        """
        self.fleet_size = fleet_size
        self.ev_capacity = ev_capacity
        self.min_charge_kw = min_charge_kw
        self.max_charge_kw = max_charge_kw
        self.min_soc_target = min_soc_target
        self.workdays_only = workdays_only

        # Minimálny výkon a krok výkonu pre IDM v MW (0.1 MW = 100 kW)
        self.idm_min_power_mw = 0.1
        self.idm_power_step_mw = 0.1

        # Úložisko údajov
        self.dam_prices = None
        self.idm_15_prices = None
        self.idm_60_prices = None
        self.price_change_probs_60 = None
        self.price_change_probs_15 = None

        # Vozidlá flotily
        self.vehicles = []

        # Údaje PMF (pravdepodobnostné rozdelenia)
        self.arrival_pmf = None
        self.departure_pmf = None
        self.soc_return_pmf = None

        # Úložisko výsledkov
        self.baseline_results = None
        self.optimized_results = None

    def validate_parameters(self):
        """Kontrola platnosti vstupných parametrov"""
        if self.fleet_size <= 0:
            raise ValueError("Veľkosť flotily musí byť kladné číslo")

        if self.ev_capacity <= 0:
            raise ValueError("Kapacita batérie musí byť kladné číslo")

        if self.min_charge_kw < 0 or self.max_charge_kw <= 0 or self.min_charge_kw > self.max_charge_kw:
            raise ValueError("Neplatné hodnoty nabíjacieho výkonu")

        if self.min_soc_target <= 0 or self.min_soc_target > 1:
            raise ValueError("Cieľový SOC musí byť v rozmedzí (0, 1]")

    def load_market_data(self, data_dir):
        """Načítanie trhových údajov z Excel súborov"""
        # Načítanie DAM cien
        dam_file = os.path.join(data_dir, 'DAM_2024v2.xlsx')
        self.dam_prices = self._load_dam_data(dam_file)

        # Načítanie IDM cien
        idm_file = os.path.join(data_dir, 'IDM_2024v2.xlsx')
        self.idm_15_prices, self.idm_60_prices = self._load_idm_data(idm_file)

        # Načítanie pravdepodobností zmien cien
        prob_60_file = os.path.join(data_dir, 'weekend_workday_price_change_probabilities.csv')
        prob_15_file = os.path.join(data_dir, 'weighted_weekend_workday_probabilities_15.csv')

        self.price_change_probs_60 = pd.read_csv(prob_60_file, index_col=[0, 1])
        self.price_change_probs_15 = pd.read_csv(prob_15_file, index_col=[0, 1, 2])

    def _load_dam_data(self, file_path):
        """Načítanie DAM dát a pridanie časového indexu"""
        # Načítanie sheetu do DataFrame
        df_60dt = pd.read_excel(file_path, sheet_name='DT60')

        # Generovanie časového radu pre celý rok 2024
        start_date = '2024-01-01 00:00:00'
        end_date = '2024-12-31 23:59:59'

        # Spracovanie sheetu "DT60"
        time_series_60 = pd.date_range(start=start_date, end=end_date, freq='H')  # 1-hodinové intervaly
        df_60dt['Time'] = time_series_60[:len(df_60dt)]  # Zabezpečenie, aby časový rad zodpovedal dĺžke DataFrame
        df_60dt.rename(columns={df_60dt.columns[0]: 'cena'}, inplace=True)  # Premenovanie prvého stĺpca na "cena"

        return df_60dt

    def _load_idm_data(self, file_path):
        """Načítanie IDM dát a pridanie časového indexu"""
        # Načítanie sheetov do DataFrames
        sheets = pd.read_excel(file_path, sheet_name=['15', '60'])

        # Generovanie časového radu pre celý rok 2024
        start_date = '2024-01-01 00:00:00'
        end_date = '2024-12-31 23:59:59'

        # Spracovanie sheetu "15"
        df_15 = sheets['15']
        time_series_15 = pd.date_range(start=start_date, end=end_date, freq='15T')  # 15-minútové intervaly
        df_15['Time'] = time_series_15[:len(df_15)]  # Zabezpečenie, aby časový rad zodpovedal dĺžke DataFrame
        df_15.rename(columns={df_15.columns[0]: 'cena'}, inplace=True)  # Premenovanie prvého stĺpca na "cena"

        # Spracovanie sheetu "60"
        df_60 = sheets['60']
        time_series_60 = pd.date_range(start=start_date, end=end_date, freq='H')  # 1-hodinové intervaly
        df_60['Time'] = time_series_60[:len(df_60)]  # Zabezpečenie, aby časový rad zodpovedal dĺžke DataFrame
        df_60.rename(columns={df_60.columns[0]: 'cena'}, inplace=True)  # Premenovanie prvého stĺpca na "cena"

        return df_15, df_60

    def set_manual_pmfs(self, arrival_probs, departure_probs, soc_ranges_probs):
        """
        Manuálne nastavenie PMF s využitím pravdepodobnostných hodnôt

        Parametre:
        -----------
        arrival_probs : dict
            Slovník s hodinami (int alebo tuple (hour, minute)) ako kľúčmi
            a pravdepodobnostnými hodnotami (0-100) ako hodnotami pre príchody vozidiel
        departure_probs : dict
            Slovník s hodinami (int alebo tuple (hour, minute)) ako kľúčmi
            a pravdepodobnostnými hodnotami (0-100) ako hodnotami pre odchody vozidiel
        soc_ranges_probs : dict
            Slovník s SOC rozsahmi ako kľúčmi (tuple of start,end)
            a pravdepodobnostnými hodnotami (0-100) ako hodnotami
        """
        # Validácia a normalizácia pravdepodobností
        arrival_probs = self._normalize_probabilities(arrival_probs)
        departure_probs = self._normalize_probabilities(departure_probs)
        soc_ranges_probs = self._normalize_probabilities(soc_ranges_probs)

        # Určíme, či prišli hodinové alebo polhodinové intervaly
        is_half_hour = False
        if any(isinstance(k, tuple) for k in arrival_probs.keys()):
            is_half_hour = True
            print("Detekované polhodinové intervaly pre PMF príchodov")

        # Vytvorenie PMF príchodov
        base_date = datetime.today().date()
        arrival_data = []

        if is_half_hour:
            # Polhodinové intervaly
            for (hour, minute), prob in arrival_probs.items():
                start_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                end_time = start_time + timedelta(minutes=29, seconds=59)
                arrival_data.append([start_time, end_time, prob])
        else:
            # Hodinové intervaly
            for hour, prob in arrival_probs.items():
                start_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=0))
                end_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=59))
                arrival_data.append([start_time, end_time, prob])

        self.arrival_pmf = pd.DataFrame(arrival_data, columns=['start_datetime', 'end_datetime', 'probability'])
        self.arrival_pmf['cum_probability'] = self.arrival_pmf['probability'].cumsum()

        # Určíme, či prišli hodinové alebo polhodinové intervaly pre odchody
        is_half_hour = False
        if any(isinstance(k, tuple) for k in departure_probs.keys()):
            is_half_hour = True
            print("Detekované polhodinové intervaly pre PMF odchodov")

        # Vytvorenie PMF odchodov
        departure_data = []

        if is_half_hour:
            # Polhodinové intervaly
            for (hour, minute), prob in departure_probs.items():
                start_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=minute))
                end_time = start_time + timedelta(minutes=29, seconds=59)
                departure_data.append([start_time, end_time, prob])
        else:
            # Hodinové intervaly
            for hour, prob in departure_probs.items():
                start_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=0))
                end_time = datetime.combine(base_date, datetime.min.time().replace(hour=hour, minute=59))
                departure_data.append([start_time, end_time, prob])

        self.departure_pmf = pd.DataFrame(departure_data, columns=['start_datetime', 'end_datetime', 'probability'])
        self.departure_pmf['cum_probability'] = self.departure_pmf['probability'].cumsum()

        # Vytvorenie PMF SOC
        soc_data = []

        for (start_soc, end_soc), prob in soc_ranges_probs.items():
            soc_data.append([start_soc, end_soc, prob])

        self.soc_return_pmf = pd.DataFrame(soc_data, columns=['start_soc', 'end_soc', 'probability'])
        self.soc_return_pmf['cum_probability'] = self.soc_return_pmf['probability'].cumsum()

        # Výpis potvrdenia
        print("Manuálne PMF boli nastavené:")
        print(f"PMF príchodov má {len(self.arrival_pmf)} časových slotov")
        print(f"PMF odchodov má {len(self.departure_pmf)} časových slotov")
        print(f"PMF SOC má {len(self.soc_return_pmf)} rozsahov")

    def _normalize_probabilities(self, prob_dict):
        """Normalizácia pravdepodobností, aby súčet bol 100"""
        total = sum(prob_dict.values())
        if total == 0:
            raise ValueError("Pravdepodobnosti majú nulový súčet")
        return {k: (v / total) * 100 for k, v in prob_dict.items()}

    def set_uniform_pmfs(self):
        """Nastavenie rovnomerných (rovnakých) pravdepodobností pre všetky hodiny a SOC rozsahy"""
        # Rovnomerné pravdepodobnosti príchodov (17:00 - 22:00)
        arrival_hourly_probs = {hour: 0 for hour in range(24)}
        for hour in range(17, 23):  # 17:00 to 22:00
            arrival_hourly_probs[hour] = 1

        # Rovnomerné pravdepodobnosti odchodov (6:00 - 9:00)
        departure_hourly_probs = {hour: 0 for hour in range(24)}
        for hour in range(6, 10):  # 6:00 to 9:00
            departure_hourly_probs[hour] = 1

        # Rovnomerné SOC rozsahy (10% to 60% v 10% krokoch)
        soc_ranges_probs = {
            (0.1, 0.2): 1,
            (0.2, 0.3): 1,
            (0.3, 0.4): 1,
            (0.4, 0.5): 1,
            (0.5, 0.6): 1
        }

        # Nastavenie PMF
        self.set_manual_pmfs(arrival_hourly_probs, departure_hourly_probs, soc_ranges_probs)

    def _sample_from_pmf(self, pmf, rand_val=None):
        """
        Vzorkovanie hodnoty z PMF s použitím náhodnej hodnoty medzi 0-100

        Parametre:
        -----------
        pmf : DataFrame
            Pravdepodobnostné rozdelenie
        rand_val : float, voliteľný
            Náhodná hodnota na použitie pri vzorkovaní. Ak nie je zadaná, vygeneruje sa nová.

        Výstup:
        -------
        Vzorkovaná hodnota (riadok z PMF DataFrame)
        """
        if rand_val is None:
            rand_val = random.uniform(0, 100)

        # Kontrola, či PMF má požadované stĺpce
        if 'cum_probability' not in pmf.columns:
            pmf['cum_probability'] = pmf['probability'].cumsum()

        # Nájdenie intervalu, do ktorého spadá náhodná hodnota
        idx = np.searchsorted(pmf['cum_probability'], rand_val)

        if idx >= len(pmf):
            idx = len(pmf) - 1

        return pmf.iloc[idx]

    def _sample_time(self, time_pmf, base_date, rand_val=None):
        """
        Vzorkovanie času z časového PMF

        Parametre:
        -----------
        time_pmf : DataFrame
            Časové pravdepodobnostné rozdelenie
        base_date : datetime
            Základný dátum na kombináciu s vygenerovaným časom
        rand_val : float, voliteľný
            Náhodná hodnota na použitie pri vzorkovaní

        Výstup:
        -------
        Vzorkovaný čas ako datetime objekt
        """
        try:
            # Pokus o vzorkovanie z PMF
            sampled_bin = self._sample_from_pmf(time_pmf, rand_val)

            # Kontrola, či máme start_datetime/end_datetime stĺpce
            if 'start_datetime' in sampled_bin and 'end_datetime' in sampled_bin:
                start_time = sampled_bin['start_datetime'].time()
                end_time = sampled_bin['end_datetime'].time()
            else:
                # Ak stĺpce chýbajú, použijeme prvé dva stĺpce
                start_time = pd.to_datetime(sampled_bin.iloc[0]).time()
                end_time = pd.to_datetime(sampled_bin.iloc[1]).time()

            # Konverzia na sekundy od polnoci
            start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
            end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

            # Vzorkovanie náhodnej sekundy v rámci rozsahu
            random_seconds = random.randint(start_seconds, end_seconds)

            # Konverzia späť na čas
            hours = random_seconds // 3600
            minutes = (random_seconds % 3600) // 60
            seconds = random_seconds % 60

            # Kombinácia so základným dátumom
            sampled_time = datetime.combine(base_date.date(), datetime.min.time().replace(
                hour=hours, minute=minutes, second=seconds))

            return sampled_time

        except Exception as e:
            # Ak vzorkovanie zlyhá, vrátime rozumný predvolený čas
            print(f"Chyba pri vzorkovaní času: {str(e)}")
            if 'departure' in str(time_pmf) or 'start_time' in str(time_pmf):
                # Ranný čas pre odchod (8:00)
                return datetime.combine(base_date.date(), datetime.min.time().replace(hour=8))
            else:
                # Večerný čas pre príchod (18:00)
                return datetime.combine(base_date.date(), datetime.min.time().replace(hour=18))

    def _sample_soc(self, rand_val=None):
        """
        Vzorkovanie hodnoty SOC z SOC PMF

        Parametre:
        -----------
        rand_val : float, voliteľný
            Náhodná hodnota na použitie pri vzorkovaní

        Výstup:
        -------
        Vzorkovaná hodnota SOC (0-1)
        """
        try:
            sampled_bin = self._sample_from_pmf(self.soc_return_pmf, rand_val)

            # Získanie náhodného SOC medzi začiatkom a koncom intervalu
            if 'start_soc' in sampled_bin and 'end_soc' in sampled_bin:
                start_soc = sampled_bin['start_soc']
                end_soc = sampled_bin['end_soc']
            else:
                # Ak stĺpce chýbajú, použijeme prvé dva stĺpce
                start_soc = float(sampled_bin.iloc[0]) / 100 if pd.notna(sampled_bin.iloc[0]) else 0.3
                end_soc = float(sampled_bin.iloc[1]) / 100 if pd.notna(sampled_bin.iloc[1]) else 0.4

            return random.uniform(start_soc, end_soc)
        except Exception as e:
            # Ak vzorkovanie zlyhá, vrátime rozumný predvolený SOC
            print(f"Chyba pri vzorkovaní SOC: {str(e)}")
            return 0.3  # 30% ako predvolená hodnota

    def initialize_fleet(self, simulation_start, simulation_days=1):
        """
        Inicializácia flotily s náhodnými časmi príchodu/odchodu a hodnotami SOC

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_days : int
            Počet dní na simuláciu
        """
        self.vehicles = []

        print(f"Inicializácia flotily s {self.fleet_size} vozidlami pre {simulation_days} dní...")
        print(f"Režim: {'Len pracovné dni' if self.workdays_only else 'Všetky dni'}")

        simulation_end = simulation_start + timedelta(days=simulation_days)

        # KĽÚČOVÁ ZMENA: Pridať vozidlá, ktoré sa nabíjajú už pred začiatkom simulácie
        # Tieto vozidlá sú už pripojené na začiatku simulácie (t.j. ich arrival_time je pred simulation_start)
        pre_simulation_vehicles_count = min(self.fleet_size // 3, 50)  # Tretina flotily alebo max 50 vozidiel

        print(f"Pridávanie {pre_simulation_vehicles_count} vozidiel, ktoré sú už pripojené na začiatku simulácie")

        for i in range(pre_simulation_vehicles_count):
            # Tieto vozidlá prišli pred začiatkom simulácie (1-6 hodín pred)
            hours_before = random.randint(1, 6)
            arrival_time = simulation_start - timedelta(hours=hours_before)

            # Odchod je v prvý deň ráno alebo doobeda
            departure_hour = random.randint(7, 12)
            departure_time = simulation_start.replace(hour=departure_hour, minute=random.randint(0, 59))

            # Nízky SOC pre tieto vozidlá, aby potrebovali nabíjanie
            return_soc = random.uniform(0.1, 0.4)  # 10-40% SOC

            # Výpočet potrebnej energie
            energy_needed_kwh = (self.min_soc_target - return_soc) * self.ev_capacity
            charging_hours = (departure_time - arrival_time).total_seconds() / 3600

            if energy_needed_kwh > 0 and charging_hours > 0:
                # Vytvorenie slovníka vozidla
                vehicle = {
                    'id': f"pre_{i}",  # ID pre predsimuláciu
                    'day': -1,  # Špeciálny identifikátor pre vozidlá pred začiatkom simulácie
                    'arrival_time': arrival_time,
                    'departure_time': departure_time,
                    'return_soc': return_soc,
                    'target_soc': self.min_soc_target,
                    'current_soc': return_soc,
                    'capacity_kwh': self.ev_capacity,
                    'min_charge_kw': self.min_charge_kw,
                    'max_charge_kw': self.max_charge_kw,
                    'charging_schedule': {},
                    'energy_needed_kwh': energy_needed_kwh,
                    'charging_hours': charging_hours
                }

                self.vehicles.append(vehicle)

        # Pokračujeme s generovaním vozidiel pre každý deň simulácie
        for day in range(simulation_days):
            day_start = simulation_start + timedelta(days=day)
            day_end = day_start + timedelta(days=1)

            # Kontrola, či je deň pracovný deň, ak je aktivovaný režim len pracovných dní
            is_weekend = day_start.weekday() >= 5  # 5=sobota, 6=nedeľa
            if self.workdays_only and is_weekend:
                print(f"Preskakujem deň {day + 1}/{simulation_days}: {day_start.date()} (víkend)")
                continue

            print(f"Generovanie vozidiel pre deň {day + 1}/{simulation_days}: {day_start.date()}"
                  f"{' (víkend)' if is_weekend else ' (pracovný deň)'}")

            # Pre prvý deň pridáme vozidlá prichádzajúce skoro ráno aj vozidlá prichádzajúce počas dňa
            if day == 0:
                # Určíme, koľko vozidiel generujeme pre každý časový blok
                morning_arrivals = self.fleet_size // 4  # 25% flotily prichádza ráno (6-10)
                day_arrivals = self.fleet_size // 4  # 25% flotily prichádza počas dňa (10-16)
                evening_arrivals = self.fleet_size // 2  # 50% flotily prichádza večer (štandardne)

                # 1. Ranné príchody (6:00 - 10:00)
                for i in range(morning_arrivals):
                    arrival_hour = random.randint(6, 10)
                    arrival_time = day_start.replace(hour=arrival_hour, minute=random.randint(0, 59))

                    # Odchod je večer alebo nasledujúci deň ráno
                    if random.random() < 0.3:  # 30% šanca na odchod v ten istý deň
                        departure_hour = random.randint(19, 23)
                        departure_time = day_start.replace(hour=departure_hour, minute=random.randint(0, 59))
                    else:  # 70% šanca na odchod nasledujúci deň
                        next_day = day_start + timedelta(days=1)
                        departure_hour = random.randint(6, 10)
                        departure_time = next_day.replace(hour=departure_hour, minute=random.randint(0, 59))

                    # Generovanie stavu nabitia a pridanie vozidla
                    return_soc = random.uniform(0.2, 0.5)  # 20-50% SOC
                    energy_needed_kwh = (self.min_soc_target - return_soc) * self.ev_capacity
                    charging_hours = (departure_time - arrival_time).total_seconds() / 3600

                    if energy_needed_kwh > 0 and charging_hours > 0:
                        vehicle = {
                            'id': f"{day}_morning_{i}",
                            'day': day,
                            'arrival_time': arrival_time,
                            'departure_time': departure_time,
                            'return_soc': return_soc,
                            'target_soc': self.min_soc_target,
                            'current_soc': return_soc,
                            'capacity_kwh': self.ev_capacity,
                            'min_charge_kw': self.min_charge_kw,
                            'max_charge_kw': self.max_charge_kw,
                            'charging_schedule': {},
                            'energy_needed_kwh': energy_needed_kwh,
                            'charging_hours': charging_hours
                        }
                        self.vehicles.append(vehicle)

                # 2. Denné príchody (10:00 - 16:00)
                for i in range(day_arrivals):
                    arrival_hour = random.randint(10, 16)
                    arrival_time = day_start.replace(hour=arrival_hour, minute=random.randint(0, 59))

                    # Odchod je večer alebo nasledujúci deň ráno
                    if random.random() < 0.4:  # 40% šanca na odchod v ten istý deň
                        departure_hour = random.randint(19, 23)
                        departure_time = day_start.replace(hour=departure_hour, minute=random.randint(0, 59))
                    else:  # 60% šanca na odchod nasledujúci deň
                        next_day = day_start + timedelta(days=1)
                        departure_hour = random.randint(6, 10)
                        departure_time = next_day.replace(hour=departure_hour, minute=random.randint(0, 59))

                    # Generovanie stavu nabitia a pridanie vozidla
                    return_soc = random.uniform(0.2, 0.5)  # 20-50% SOC
                    energy_needed_kwh = (self.min_soc_target - return_soc) * self.ev_capacity
                    charging_hours = (departure_time - arrival_time).total_seconds() / 3600

                    if energy_needed_kwh > 0 and charging_hours > 0:
                        vehicle = {
                            'id': f"{day}_day_{i}",
                            'day': day,
                            'arrival_time': arrival_time,
                            'departure_time': departure_time,
                            'return_soc': return_soc,
                            'target_soc': self.min_soc_target,
                            'current_soc': return_soc,
                            'capacity_kwh': self.ev_capacity,
                            'min_charge_kw': self.min_charge_kw,
                            'max_charge_kw': self.max_charge_kw,
                            'charging_schedule': {},
                            'energy_needed_kwh': energy_needed_kwh,
                            'charging_hours': charging_hours
                        }
                        self.vehicles.append(vehicle)

                # 3. Večerné príchody (štandardný vzor)
                remaining_vehicles = self.fleet_size - pre_simulation_vehicles_count - morning_arrivals - day_arrivals
            else:
                # Pre ostatné dni používame štandardné rozloženie
                remaining_vehicles = self.fleet_size

            # Štandardné vozidlá s príchodom podľa PMF (zvyčajne večer)
            for i in range(remaining_vehicles):
                # Vzorkovanie času príchodu pre tento deň
                arrival_time = self._sample_time(self.arrival_pmf, day_start)

                # Vzorkovanie času odchodu pre nasledujúci deň
                next_day = day_start + timedelta(days=1)

                # Ak nasledujúci deň je víkend a máme len pracovné dni, posunúť odchod na pondelok
                if self.workdays_only and next_day.weekday() >= 5:
                    days_to_add = 8 - next_day.weekday()  # 8-6=2 pre nedeľu, 8-5=3 pre sobotu
                    next_day = next_day + timedelta(days=days_to_add)

                departure_time = self._sample_time(self.departure_pmf, next_day)

                # Zabezpečenie, aby odchod bol po príchode
                if departure_time <= arrival_time:
                    departure_time = arrival_time + timedelta(hours=6)  # Aspoň 6 hodín na nabíjanie

                # Vzorkovanie SOC
                return_soc = self._sample_soc()

                # Výpočet potrebnej energie
                energy_needed_kwh = (self.min_soc_target - return_soc) * self.ev_capacity
                charging_hours = (departure_time - arrival_time).total_seconds() / 3600

                # Pridanie vozidla, iba ak potrebuje nabíjanie a má na to čas
                if energy_needed_kwh > 0 and charging_hours > 0:
                    vehicle = {
                        'id': f"{day}_evening_{i}",
                        'day': day,
                        'arrival_time': arrival_time,
                        'departure_time': departure_time,
                        'return_soc': return_soc,
                        'target_soc': self.min_soc_target,
                        'current_soc': return_soc,
                        'capacity_kwh': self.ev_capacity,
                        'min_charge_kw': self.min_charge_kw,
                        'max_charge_kw': self.max_charge_kw,
                        'charging_schedule': {},
                        'energy_needed_kwh': energy_needed_kwh,
                        'charging_hours': charging_hours
                    }

                    self.vehicles.append(vehicle)

        # Zoradenie vozidiel podľa času príchodu
        self.vehicles.sort(key=lambda x: x['arrival_time'])

        # Výpis štatistík flotily
        print(f"Flotila inicializovaná s {len(self.vehicles)} vozidlami.")
        if self.vehicles:
            print(
                f"Priemerná potrebná energia: {sum(v['energy_needed_kwh'] for v in self.vehicles) / len(self.vehicles):.2f} kWh")
            print(
                f"Priemerné nabíjacie okno: {sum(v['charging_hours'] for v in self.vehicles) / len(self.vehicles):.2f} hodín")

            # Počítanie vozidiel podľa častí dňa a dní
            time_categories = {'Pred simuláciou': 0, 'Ráno': 0, 'Deň': 0, 'Večer': 0}
            vehicles_per_day = {}

            for v in self.vehicles:
                # Kategorizácia podľa času príchodu
                if v['day'] == -1:
                    time_categories['Pred simuláciou'] += 1
                else:
                    arrival_hour = v['arrival_time'].hour
                    if 6 <= arrival_hour < 10:
                        time_categories['Ráno'] += 1
                    elif 10 <= arrival_hour < 16:
                        time_categories['Deň'] += 1
                    else:
                        time_categories['Večer'] += 1

                # Kategorizácia podľa dňa
                day = v['day']
                if day >= 0:  # Ignorujeme vozidlá pred simuláciou pri počítaní podľa dní
                    vehicles_per_day[day] = vehicles_per_day.get(day, 0) + 1

            print("\nVozidlá podľa času príchodu:")
            for category, count in time_categories.items():
                print(f"  {category}: {count} vozidiel")

            print("\nVozidlá podľa dní:")
            for day in sorted(vehicles_per_day.keys()):
                day_date = (simulation_start + timedelta(days=day)).date()
                print(f"  Deň {day + 1} ({day_date}): {vehicles_per_day[day]} vozidiel")

            # Výpis vozidiel z prvej noci
            print("\nVozidlá nabíjajúce sa počas prvej noci:")
            first_night_vehicles = [v for v in self.vehicles if
                                    (v['day'] == -1 or  # Pred simuláciou
                                     (v['day'] == 0 and v['arrival_time'].hour < 6)  # Alebo skoro ráno prvého dňa
                                     ) and
                                    v['departure_time'] > simulation_start]

            for i, v in enumerate(first_night_vehicles[:5]):  # Zobrazíme prvých 5
                print(f"Vozidlo {v['id']}: Príchod {v['arrival_time']}, Odchod {v['departure_time']}, "
                      f"SOC {v['return_soc'] * 100:.1f}%, Potrebuje {v['energy_needed_kwh']:.2f} kWh")

            if len(first_night_vehicles) > 5:
                print(f"  ... a ďalších {len(first_night_vehicles) - 5} vozidiel")

            # Výpis vzorky všetkých vozidiel pre overenie
            print("\nVzorka všetkých vozidiel:")
            for i in range(min(5, len(self.vehicles))):
                v = self.vehicles[i]
                day_label = "Pred simuláciou" if v['day'] == -1 else f"Deň {v['day'] + 1}"
                print(f"Vozidlo {v['id']}: {day_label}, Príchod {v['arrival_time']}, Odchod {v['departure_time']}, "
                      f"SOC {v['return_soc'] * 100:.1f}%, Potrebuje {v['energy_needed_kwh']:.2f} kWh")
        else:
            print("Varovanie: Neboli vytvorené žiadne vozidlá! Skontrolujte PMF dáta a parametre vozidiel.")

    def get_eligible_vehicles(self, start_time, end_time):
        """Získanie vozidiel relevantných pre dané časové obdobie"""
        return [v for v in self.vehicles
                if v['departure_time'] > start_time and v['arrival_time'] < end_time]

    def reset_vehicle_states(self):
        """Reset stavov všetkých vozidiel na počiatočné hodnoty"""
        for vehicle in self.vehicles:
            vehicle['current_soc'] = vehicle['return_soc']
            vehicle['charging_schedule'] = {}

    def calculate_fleet_energy_needs(self):
        """Výpočet celkovej energetickej potreby flotily"""
        total_energy_needed = 0
        for vehicle in self.vehicles:
            energy_needed = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']
            if energy_needed > 0:
                total_energy_needed += energy_needed
        return total_energy_needed

    def estimate_future_prices(self, current_time, forecast_horizon=24, interval=60):
        """
        Odhad budúcich cien na základe historických pravdepodobností

        Parametre:
        -----------
        current_time : datetime
            Aktuálny čas, od ktorého sa má predpovedať
        forecast_horizon : int
            Počet hodín na predpoveď
        interval : int
            Časový interval v minútach (15 alebo 60)

        Výstup:
        --------
        DataFrame s odhadovanými cenami
        """
        # Určenie, či ide o víkend alebo pracovný deň
        is_weekend = current_time.weekday() >= 5
        day_type = 'Weekend' if is_weekend else 'Workday'

        # Získanie aktuálnej ceny
        if interval == 15:
            # Nájdenie najbližšieho času v 15-minútových údajoch
            idx = (self.idm_15_prices['Time'] - current_time).abs().idxmin()
            if idx is not None and idx < len(self.idm_15_prices):
                current_price = self.idm_15_prices.iloc[idx]['cena']
            else:
                current_price = 50.0  # Predvolená cena, ak údaje nie sú k dispozícii
            price_data = self.idm_15_prices
            probs = self.price_change_probs_15
        else:  # 60-minútový interval
            # Nájdenie najbližšieho času v 60-minútových údajoch
            idx = (self.idm_60_prices['Time'] - current_time).abs().idxmin()
            if idx is not None and idx < len(self.idm_60_prices):
                current_price = self.idm_60_prices.iloc[idx]['cena']
            else:
                current_price = 50.0  # Predvolená cena, ak údaje nie sú k dispozícii
            price_data = self.idm_60_prices
            probs = self.price_change_probs_60

        # Generovanie časov predpovede
        if interval == 15:
            forecast_times = pd.date_range(start=current_time, periods=forecast_horizon * 4, freq='15T')
        else:
            forecast_times = pd.date_range(start=current_time, periods=forecast_horizon, freq='H')

        # Inicializácia predpovede s aktuálnou cenou
        forecast = pd.DataFrame({
            'Time': forecast_times,
            'estimated_price': current_price
        })

        # Aktualizácia každého časového kroku na základe pravdepodobností
        for i in range(1, len(forecast)):
            prev_price = forecast.iloc[i - 1]['estimated_price']

            # Získanie hodiny a minúty
            hour = forecast.iloc[i]['Time'].hour

            if interval == 15:
                minute = forecast.iloc[i]['Time'].minute
                # Získanie pravdepodobností pre túto hodinu a minútu
                try:
                    prob_idx = (day_type, hour, minute)
                    hour_probs = probs.loc[prob_idx]
                except (KeyError, TypeError) as e:
                    # Ak sa nenájde presná zhoda, použijeme predvolené pravdepodobnosti
                    hour_probs = pd.Series({'Increase': 33.3, 'Decrease': 33.3, 'No Change': 33.4})
            else:
                # Získanie pravdepodobností pre túto hodinu
                try:
                    prob_idx = (day_type, hour)
                    hour_probs = probs.loc[prob_idx]
                except (KeyError, TypeError) as e:
                    # Ak sa nenájde presná zhoda, použijeme predvolené pravdepodobnosti
                    hour_probs = pd.Series({'Increase': 33.3, 'Decrease': 33.3, 'No Change': 33.4})

            # Vzorkovanie z rozdelenia pravdepodobnosti
            rand = random.uniform(0, 100)
            cumsum = 0
            direction = 'No Change'

            for dir_name, prob in hour_probs.items():
                cumsum += prob
                if rand <= cumsum:
                    direction = dir_name
                    break

            # Aplikácia zmeny ceny
            if direction == 'Increase':
                # Náhodné zvýšenie medzi 1% a 5%
                change_factor = 1 + random.uniform(0.01, 0.05)
            elif direction == 'Decrease':
                # Náhodné zníženie medzi 1% a 5%
                change_factor = 1 - random.uniform(0.01, 0.05)
            else:  # No Change
                change_factor = 1

            # Aktualizácia ceny
            forecast.iloc[i, forecast.columns.get_loc('estimated_price')] = prev_price * change_factor

        return forecast

    def baseline_strategy(self, simulation_start, simulation_end=None):
        """
        Implementácia základnej stratégie nabíjania (iba DAM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.

        Výstup:
        --------
        DataFrame so základným nabíjacím harmonogramom a nákladmi
        """
        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        simulation_days = (simulation_end - simulation_start).total_seconds() / (24 * 3600)
        simulation_days = max(1, round(simulation_days))

        print(f"Spúšťanie základnej stratégie od {simulation_start} do {simulation_end} ({simulation_days} dní)")

        # Výpočet počtu hodín v simulácii
        simulation_hours = int((simulation_end - simulation_start).total_seconds() / 3600)
        if simulation_hours <= 0:
            simulation_hours = 24  # Predvolene 24 hodín

        print(f"Vytváranie harmonogramu s {simulation_hours} hodinovými periódami")

        # Vytvorenie harmonogramu s hodinovými intervalmi pre celé simulačné obdobie
        schedule_times = pd.date_range(
            start=simulation_start,
            periods=simulation_hours,
            freq='H'
        )

        baseline = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_price': 0.0,
            'dam_cost': 0.0,
            'total_ev_charging': 0
        })

        # Debugovacie informácie
        print(f"Nájdených {len(self.vehicles)} vozidiel pre simulačné obdobie")
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"{len(eligible_vehicles)} vozidiel je vhodných na nabíjanie v tomto období")

        # Analýza rozloženia vozidiel podľa dní
        days_covered = {}
        for v in eligible_vehicles:
            arrival_day = (v['arrival_time'].date() - simulation_start.date()).days
            days_covered[arrival_day] = days_covered.get(arrival_day, 0) + 1

        print(f"Vozidlá pokrývajú {len(days_covered)} dní v simulačnom období")
        for day, count in sorted(days_covered.items()):
            day_date = (simulation_start + timedelta(days=day)).date()
            print(f"  Deň {day + 1} ({day_date}): {count} vhodných vozidiel")

        # Získanie DAM cien pre simulačné obdobie
        dam_period_prices = self.dam_prices.copy()

        print(f"Nájdených {len(dam_period_prices)} DAM cenových záznamov")

        # Mapovanie DAM cien na harmonogram
        for idx, row in baseline.iterrows():
            time = row['Time']

            # Nájdenie DAM ceny pre túto hodinu
            dam_prices_in_range = dam_period_prices[
                (dam_period_prices['Time'] >= time) &
                (dam_period_prices['Time'] < time + timedelta(hours=1))
                ]

            if not dam_prices_in_range.empty:
                baseline.at[idx, 'dam_price'] = dam_prices_in_range.iloc[0]['cena']
            else:
                # Nájdenie najbližšej ceny
                closest_idx = (dam_period_prices['Time'] - time).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(dam_period_prices):
                    baseline.at[idx, 'dam_price'] = dam_period_prices.iloc[closest_idx]['cena']
                else:
                    # Predvolená hodnota
                    baseline.at[idx, 'dam_price'] = 50.0  # Predvolená hodnota, ak sa nenájde zhoda

        # Počítanie nabíjaných vozidiel
        charged_vehicles = 0

        # Pre každé vhodné vozidlo určíme jeho nabíjacie potreby a harmonogram
        for vehicle in eligible_vehicles:
            # Výpočet potrebnej energie
            energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']

            if energy_needed_kwh <= 0:
                continue  # Vozidlo nepotrebuje nabíjanie

            # Nájdenie nabíjacieho okna
            charging_start = max(vehicle['arrival_time'], baseline['Time'].min())
            charging_end = min(vehicle['departure_time'], baseline['Time'].max())

            # Získanie všetkých časových slotov v nabíjacom okne
            charging_slots = baseline[
                (baseline['Time'] >= charging_start) &
                (baseline['Time'] < charging_end)
                ]

            if len(charging_slots) == 0:
                print(f"Varovanie: Žiadne platné nabíjacie sloty pre vozidlo {vehicle['id']}")
                continue  # Žiadne platné nabíjacie sloty

            # Zoradenie podľa ceny pre nabíjanie v najlacnejších hodinách
            charging_slots = charging_slots.sort_values('dam_price').copy()

            # Alokácia nabíjacieho výkonu
            remaining_energy = energy_needed_kwh
            vehicle_schedule = {}
            hours_charged = 0

            for _, slot in charging_slots.iterrows():
                if remaining_energy <= 0:
                    break

                time = slot['Time']

                # Výpočet maximálnej energie, ktorú možno nabiť v tejto hodinovej perióde
                max_energy_per_hour = min(vehicle['max_charge_kw'], remaining_energy)

                # Aktualizácia harmonogramu
                baseline.loc[baseline['Time'] == time, 'total_charge_kw'] += max_energy_per_hour
                baseline.loc[baseline['Time'] == time, 'total_ev_charging'] += 1

                # Aktualizácia nabíjacieho plánu vozidla
                vehicle_schedule[time] = max_energy_per_hour

                # Aktualizácia zostávajúcej energie
                remaining_energy -= max_energy_per_hour
                hours_charged += 1

            # Uloženie nabíjacieho plánu do vozidla
            vehicle['charging_schedule'] = vehicle_schedule

            # Aktualizácia aktuálneho SOC vozidla
            energy_charged = energy_needed_kwh - remaining_energy
            vehicle['current_soc'] = vehicle['return_soc'] + (energy_charged / vehicle['capacity_kwh'])

            if hours_charged > 0:
                charged_vehicles += 1

        # Výpočet nákladov
        baseline['dam_cost'] = baseline['total_charge_kw'] * baseline['dam_price'] / 1000  # Konverzia na MWh náklady

        # Výpočet celkovej energie a nákladov
        total_energy = baseline['total_charge_kw'].sum()  # kWh
        total_cost = baseline['dam_cost'].sum()  # €

        print(f"\nSúhrn základného nabíjania pre {simulation_days} dní:")
        print(f"Celkový počet nabíjaných vozidiel: {charged_vehicles}")
        print(f"Celková nabitá energia: {total_energy:.2f} kWh")
        print(f"Celkové náklady: {total_cost:.2f} €")
        if total_energy > 0:
            print(f"Priemerná cena: {(total_cost / total_energy * 1000):.2f} €/MWh")
        else:
            print("Priemerná cena: N/A (žiadna nabitá energia)")

        # Uloženie výsledkov do atribútu triedy
        self.baseline_results = baseline

        # Vrátenie výsledkov
        return baseline

    def optimized_strategy_OLD(self, simulation_start, simulation_end=None, options=None,DAM_ALLOCATION=0.4):
        """
        Implementácia optimalizovanej stratégie nabíjania (DAM + IDM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.
        options : dict, voliteľné
            Slovník optimalizačných možností:
            - 'analyze_price_patterns': bool - Analýza hodinových cenových vzorov
            - 'general_idm_discount': float - Aplikácia všeobecnej zľavy na IDM ceny
            - 'targeted_night_discount': float - Aplikácia cielenej zľavy počas nočných hodín

        Výstup:
        --------
        DataFrame s optimalizovaným nabíjacím harmonogramom a nákladmi
        """
        # Nastavenie predvolených možností, ak nie sú poskytnuté - vypnute defaultne nepouživat
        if options is None:
            options = {
                'analyze_price_patterns': False    ,  # Analýza cenových vzorov
                'general_idm_discount': 1,  # 5% zľava
                'targeted_night_discount': 1  # 15% zľava počas nočných hodín
            }

        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        simulation_days = (simulation_end - simulation_start).total_seconds() / (24 * 3600)
        simulation_days = max(1, round(simulation_days))

        print(f"Spúšťanie optimalizovanej stratégie od {simulation_start} do {simulation_end} ({simulation_days} dní)")
        print(f"Optimalizačné možnosti: {options}")

        # Výpočet počtu 15-minútových periód v simulácii
        simulation_periods = int((simulation_end - simulation_start).total_seconds() / 900)  # 900 sekúnd = 15 minút
        if simulation_periods <= 0:
            simulation_periods = 96  # Predvolene 24 hodín (96 periód po 15 minút)

        print(f"Vytváranie harmonogramu s {simulation_periods} 15-minútovými periódami")

        # Vytvorenie harmonogramu s 15-minútovými intervalmi pre celé simulačné obdobie
        schedule_times = pd.date_range(
            start=simulation_start,
            periods=simulation_periods,
            freq='15T'
        )

        optimized = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_charge_kw': 0.0,
            'idm_charge_kw': 0.0,
            'idm_15_charge_kw': 0.0,  # Nový stĺpec pre 15-min IDM
            'idm_60_charge_kw': 0.0,  # Nový stĺpec pre 60-min IDM
            'dam_price': 0.0,
            'idm_price': 0.0,
            'idm_15_price': 0.0,  # Cena pre 15-min IDM
            'idm_60_price': 0.0,  # Cena pre 60-min IDM
            'original_idm_15_price': 0.0,  # Na uloženie pôvodných cien pred zľavami
            'original_idm_60_price': 0.0,  # Na uloženie pôvodných cien pred zľavami
            'dam_cost': 0.0,
            'idm_cost': 0.0,
            'idm_15_cost': 0.0,  # Náklady pre 15-min IDM
            'idm_60_cost': 0.0,  # Náklady pre 60-min IDM
            'total_ev_charging': 0,
            'total_ev_charging_dam': 0,
            'total_ev_charging_idm': 0,
            'total_ev_charging_idm_15': 0,  # Vozidlá používajúce 15-min IDM
            'total_ev_charging_idm_60': 0  # Vozidlá používajúce 60-min IDM
        })

        # Debugovacie informácie
        print(f"Nájdených {len(self.vehicles)} vozidiel pre simulačné obdobie")
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"{len(eligible_vehicles)} vozidiel je vhodných na nabíjanie v tomto období")

        # Analýza rozloženia vozidiel podľa dní
        days_covered = {}
        for v in eligible_vehicles:
            arrival_day = (v['arrival_time'].date() - simulation_start.date()).days
            days_covered[arrival_day] = days_covered.get(arrival_day, 0) + 1

        print(f"Vozidlá pokrývajú {len(days_covered)} dní v simulačnom období")
        for day, count in sorted(days_covered.items()):
            day_date = (simulation_start + timedelta(days=day)).date()
            print(f"  Deň {day + 1} ({day_date}): {count} vhodných vozidiel")

        # Získanie DAM cien pre simulačné obdobie
        dam_period_prices = self.dam_prices.copy()

        # Získanie IDM cien pre simulačné obdobie - 15-minútové a 60-minútové ceny
        idm_15_period_prices = self.idm_15_prices.copy()
        idm_60_period_prices = self.idm_60_prices.copy()

        print(f"Nájdených {len(dam_period_prices)} DAM cenových záznamov, "
              f"{len(idm_15_period_prices)} IDM 15-min cenových záznamov a "
              f"{len(idm_60_period_prices)} IDM 60-min cenových záznamov")

        # Mapovanie DAM a IDM cien na harmonogram
        for idx, row in optimized.iterrows():
            time = row['Time']
            hour_start = time.replace(minute=0, second=0, microsecond=0)

            # Nájdenie DAM ceny (najbližšia hodinová zhoda)
            dam_prices_in_range = dam_period_prices[
                (dam_period_prices['Time'] >= hour_start) &
                (dam_period_prices['Time'] < hour_start + timedelta(hours=1))
                ]

            if not dam_prices_in_range.empty:
                optimized.at[idx, 'dam_price'] = dam_prices_in_range.iloc[0]['cena']
            else:
                # Nájdenie najbližšej ceny
                closest_idx = (dam_period_prices['Time'] - hour_start).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(dam_period_prices):
                    optimized.at[idx, 'dam_price'] = dam_period_prices.iloc[closest_idx]['cena']
                else:
                    # Predvolená hodnota
                    optimized.at[idx, 'dam_price'] = 50.0  # Predvolená hodnota, ak sa nenájde zhoda

            # Nájdenie IDM 15-min ceny (najbližšia 15-minútová zhoda)
            idm_15_prices_in_range = idm_15_period_prices[
                (idm_15_period_prices['Time'] >= time) &
                (idm_15_period_prices['Time'] < time + timedelta(minutes=15))
                ]

            if not idm_15_prices_in_range.empty:
                optimized.at[idx, 'idm_15_price'] = idm_15_prices_in_range.iloc[0]['cena']
                optimized.at[idx, 'original_idm_15_price'] = idm_15_prices_in_range.iloc[0]['cena']
            else:
                # Nájdenie najbližšej ceny
                closest_idx = (idm_15_period_prices['Time'] - time).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(idm_15_period_prices):
                    optimized.at[idx, 'idm_15_price'] = idm_15_period_prices.iloc[closest_idx]['cena']
                    optimized.at[idx, 'original_idm_15_price'] = idm_15_period_prices.iloc[closest_idx]['cena']
                else:
                    # Použitie DAM ceny ako zálohy
                    optimized.at[idx, 'idm_15_price'] = optimized.at[idx, 'dam_price']
                    optimized.at[idx, 'original_idm_15_price'] = optimized.at[idx, 'dam_price']

            # Nájdenie IDM 60-min ceny (najbližšia 60-minútová zhoda)
            idm_60_prices_in_range = idm_60_period_prices[
                (idm_60_period_prices['Time'] >= hour_start) &
                (idm_60_period_prices['Time'] < hour_start + timedelta(hours=1))
                ]

            if not idm_60_prices_in_range.empty:
                optimized.at[idx, 'idm_60_price'] = idm_60_prices_in_range.iloc[0]['cena']
                optimized.at[idx, 'original_idm_60_price'] = idm_60_prices_in_range.iloc[0]['cena']
            else:
                # Nájdenie najbližšej ceny
                closest_idx = (idm_60_period_prices['Time'] - hour_start).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(idm_60_period_prices):
                    optimized.at[idx, 'idm_60_price'] = idm_60_period_prices.iloc[closest_idx]['cena']
                    optimized.at[idx, 'original_idm_60_price'] = idm_60_period_prices.iloc[closest_idx]['cena']
                else:
                    # Použitie DAM ceny ako zálohy
                    optimized.at[idx, 'idm_60_price'] = optimized.at[idx, 'dam_price']
                    optimized.at[idx, 'original_idm_60_price'] = optimized.at[idx, 'dam_price']

        # MOŽNOSŤ : Analýza cenových vzorov
        best_idm_hours = []
        if options.get('analyze_price_patterns', False):
            # Nájdenie najlepších časov na použitie IDM vs DAM
            hourly_price_advantage = []
            for hour in range(24):
                hour_data = optimized[optimized['Time'].dt.hour == hour]
                if not hour_data.empty:
                    dam_prices = hour_data['dam_price'].mean()
                    idm_15_prices = hour_data['idm_15_price'].mean()
                    idm_60_prices = hour_data['idm_60_price'].mean()
                    idm_prices = hour_data[['idm_15_price', 'idm_60_price']].min(axis=1).mean()
                    advantage = dam_prices - idm_prices
                    hourly_price_advantage.append((hour, advantage))

            # Zoradenie hodín podľa cenovej výhody (najlepšie IDM hodiny najskôr)
            hourly_price_advantage.sort(key=lambda x: x[1], reverse=True)
            best_idm_hours = [h for h, adv in hourly_price_advantage if adv > 0]

            print(f"Najlepšie hodiny pre využitie IDM: {best_idm_hours}")
            print("Hodinová cenová výhoda (DAM - IDM):")
            for hour, advantage in hourly_price_advantage:
                print(f"  Hodina {hour}: {advantage:.2f} €/MWh")

        # MOŽNOSŤ : Aplikácia všeobecnej zľavy na IDM ceny
        general_discount = options.get('general_idm_discount', 1.0)
        if general_discount < 1.0:
            print(f"Aplikácia všeobecného diskontného faktora IDM: {general_discount:.2f}")
            optimized['idm_15_price'] = optimized['idm_15_price'] * general_discount
            optimized['idm_60_price'] = optimized['idm_60_price'] * general_discount

        # MOŽNOSŤ : Aplikácia cielenej zľavy počas nočných hodín
        night_discount = options.get('targeted_night_discount', 1.0)
        if night_discount < 1.0:
            print(f"Aplikácia cieleného nočného diskontného faktora: {night_discount:.2f}")
            # Aplikácia zľavy počas nočných hodín (22:00 - 06:00)
            night_mask = (optimized['Time'].dt.hour >= 22) | (optimized['Time'].dt.hour < 6)
            optimized.loc[night_mask, 'idm_15_price'] = optimized.loc[night_mask, 'idm_15_price'] * night_discount
            optimized.loc[night_mask, 'idm_60_price'] = optimized.loc[night_mask, 'idm_60_price'] * night_discount

        # Výber lepšej IDM ceny (minimum z 15-min a 60-min)
        optimized['idm_price'] = optimized[['idm_15_price', 'idm_60_price']].min(axis=1)

        # Spracovanie každého vozidla pre alokáciu nabíjania
        vehicles_dam_charged = 0
        vehicles_idm_15_charged = 0
        vehicles_idm_60_charged = 0

        # Najprv alokujeme DAM nabíjanie pre základné zaťaženie
        print("Alokácia DAM nabíjania...")
        for vehicle in eligible_vehicles:
            # Výpočet potrebnej energie
            energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']

            if energy_needed_kwh <= 0:
                continue  # Vozidlo nepotrebuje nabíjanie

            # Nájdenie nabíjacieho okna
            charging_start = max(vehicle['arrival_time'], optimized['Time'].min())
            charging_end = min(vehicle['departure_time'], optimized['Time'].max())

            # Získanie všetkých časových slotov v nabíjacom okne
            charging_slots = optimized[
                (optimized['Time'] >= charging_start) &
                (optimized['Time'] < charging_end)
                ]

            if len(charging_slots) == 0:
                print(f"Varovanie: Žiadne platné nabíjacie sloty pre vozidlo {vehicle['id']}")
                continue  # Žiadne platné nabíjacie sloty

            # MOŽNOSŤ : Úprava DAM alokácie na základe najlepších hodín
            if options.get('analyze_price_patterns', False) and best_idm_hours:
                # Použitie variabilnej DAM alokácie na základe hodiny
                dam_allocations = []
                for _, slot in charging_slots.iterrows():
                    hour = slot['Time'].hour
                    # Nižšia DAM alokácia v hodinách, kde je IDM lepší
                    if hour in best_idm_hours:
                        dam_allocations.append((slot['Time'], 0.7))  # 30% DAM v dobrých IDM hodinách
                    else:
                        dam_allocations.append((slot['Time'], 0.7))  # 70% DAM v ostatných hodinách

                # Zoradenie podľa ceny a alokácia DAM nabíjania
                dam_allocations.sort(
                    key=lambda x: (optimized.loc[optimized['Time'] == x[0], 'dam_price'].iloc[0], -x[1]))
            else:
                # Použitie fixnej 50% DAM alokácie pre všetky sloty
                dam_allocations = [(slot['Time'], DAM_ALLOCATION) for _, slot in charging_slots.iterrows()]
                dam_allocations.sort(key=lambda x: optimized.loc[optimized['Time'] == x[0], 'dam_price'].iloc[0])

            # Výpočet celkovej energie na alokáciu do DAM
            total_dam_allocation = sum(allocation for _, allocation in dam_allocations) * energy_needed_kwh / len(
                dam_allocations)
            remaining_dam_energy = total_dam_allocation
            vehicle_schedule = {}
            periods_charged = 0

            # Alokácia DAM nabíjania
            for time, _ in dam_allocations:
                if remaining_dam_energy <= 0:
                    break

                # Výpočet maximálnej energie, ktorú možno nabiť v tejto 15-minútovej perióde
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minút = 0.25 hodiny
                charge_amount = min(max_energy_per_period, remaining_dam_energy)
                charge_kw = charge_amount / 0.25  # Konverzia späť na kW

                if charge_kw > 0:
                    # Aktualizácia harmonogramu
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging'] += 1
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Aktualizácia nabíjacieho plánu vozidla
                    vehicle_schedule[time] = {'source': 'DAM', 'kw': charge_kw}

                    # Aktualizácia zostávajúcej energie
                    remaining_dam_energy -= charge_amount
                    periods_charged += 1

            # Aktualizácia aktuálneho SOC vozidla na základe DAM nabíjania
            energy_charged_dam = total_dam_allocation - remaining_dam_energy
            vehicle['current_soc'] = vehicle['return_soc'] + (energy_charged_dam / vehicle['capacity_kwh'])

            # Uloženie DAM časti nabíjacieho plánu
            vehicle['charging_schedule'] = vehicle_schedule

            if periods_charged > 0:
                vehicles_dam_charged += 1

        print(f"DAM nabíjanie alokované pre {vehicles_dam_charged} vozidiel")

        # Teraz optimalizujeme zostávajúce nabíjacie potreby pomocou IDM
        # Spracovanie časových slotov v opačnom poradí (neskoršie hodiny najskôr) na uprednostnenie neskoršieho nabíjania
        print("Alokácia IDM nabíjania...")

        # Najprv zoradíme časové sloty od najneskorších po najskoršie
        time_slots = sorted(optimized['Time'].unique(), reverse=True)

        # # Spracovanie každého časového slotu pre IDM nabíjanie
        # for time in time_slots:
        #     # Získanie riadku pre tento čas
        #     row = optimized.loc[optimized['Time'] == time].iloc[0]
        #
        #     # Total DAM power already allocated
        #     total_dam_kw = row['dam_charge_kw']
        #
        #     # FIXED: Calculate remaining capacity with the correct fleet size factor
        #     # This ensures we consider the entire fleet capacity
        #     remaining_capacity = self.fleet_size * self.max_charge_kw - total_dam_kw
        #
        #     # FIXED: Debug output to verify capacity calculation
        #     if time.hour == 18 and time.minute == 0:  # Example check at 6 PM
        #         print(f"DEBUG at {time}: Fleet size={self.fleet_size}, Max charge={self.max_charge_kw}")
        #         print(f"Total DAM kW={total_dam_kw}, Remaining capacity={remaining_capacity}")
        #
        #     # MODIFIED: Relaxed condition - Pokračujeme, if we have at least 100 kW capacity (0.1 MW)
        #     if remaining_capacity >= 100:
        #         # Zaokrúhlenie nadol na najbližších 100 kW (0.1 MW krok)
        #         idm_potential_kw = int(remaining_capacity / 100) * 100
        #
        #         # Debug output
        #         if time.hour == 18 and time.minute == 0:  # Example check at 6 PM
        #             print(f"IDM potential kW={idm_potential_kw}")
        #
        #         # Určenie, ktorý IDM trh použiť (15-min alebo 60-min)
        #         use_15_min_idm = row['idm_15_price'] <= row['idm_60_price']
        #         idm_price = row['idm_15_price'] if use_15_min_idm else row['idm_60_price']
        #
        #         # FIXED: Adjust the price comparison to be more lenient
        #         # Allow IDM even if price is slightly higher (within 5%)
        #         hour = time.hour
        #         is_best_idm_hour = hour in best_idm_hours if options.get('analyze_price_patterns', False) else False
        #
        #         # MODIFIED: More lenient condition for using IDM
        #         should_use_idm = (idm_price <= row['dam_price'] * 1.05) or is_best_idm_hour
        #
        #         if should_use_idm:
        #             # Find all vehicles needing charge at this time
        #             vehicles_needing_charge = [v for v in eligible_vehicles
        #                                        if v['current_soc'] < v['target_soc'] and
        #                                        time >= v['arrival_time'] and
        #                                        time < v['departure_time']]
        #
        #             # Sort vehicles by charging need (descending - priority for vehicles needing more charging)
        #             vehicles_needing_charge.sort(key=lambda v: v['target_soc'] - v['current_soc'], reverse=True)
        #
        #             # Allocate IDM charging
        #             idm_allocated_kw = 0
        #
        #             for vehicle in vehicles_needing_charge:
        #                 if idm_allocated_kw >= idm_potential_kw:
        #                     break
        #
        #                 # Calculate energy needed
        #                 remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle[
        #                     'capacity_kwh']
        #
        #                 if remaining_energy_needed <= 0:
        #                     continue
        #
        #                 # Calculate how much power we can charge in this 15-minute period
        #                 max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minutes = 0.25 hour
        #                 charge_amount = min(max_energy_per_period, remaining_energy_needed)
        #                 charge_kw = charge_amount / 0.25  # Convert back to kW
        #
        #                 # Calculate how much of this can be allocated to IDM (in 100 kW steps)
        #                 # Ensure we don't exceed the remaining IDM capacity
        #                 max_possible_idm_kw = min(charge_kw, idm_potential_kw - idm_allocated_kw)
        #                 idm_charge_kw = int(max_possible_idm_kw / 100) * 100  # Round down to nearest 100 kW step
        #
        #                 # Calculate the remaining power that needs to be allocated to DAM
        #                 dam_charge_kw = charge_kw - idm_charge_kw
        #
        #                 # Update the vehicle's charging and SOC only if we're allocating power
        #                 energy_charged = 0
        #
        #                 # Process IDM portion if any
        #                 if idm_charge_kw > 0:
        #                     # Update the schedule based on which IDM market we're using
        #                     if use_15_min_idm:
        #                         optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] += idm_charge_kw
        #                         optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] += 1
        #                         vehicles_idm_15_charged += 1
        #                     else:
        #                         optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] += idm_charge_kw
        #                         optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] += 1
        #                         vehicles_idm_60_charged += 1
        #
        #                     # Update total IDM and total charging
        #                     optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] += idm_charge_kw
        #                     optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += idm_charge_kw
        #                     optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] += 1
        #
        #                     # Calculate energy charged through IDM
        #                     idm_energy_charged = idm_charge_kw * 0.25  # kWh
        #                     energy_charged += idm_energy_charged
        #
        #                     # Update the allocated IDM power
        #                     idm_allocated_kw += idm_charge_kw
        #
        #                 # Process DAM portion if any
        #                 if dam_charge_kw > 0:
        #                     # Update DAM charges
        #                     optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += dam_charge_kw
        #                     optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += dam_charge_kw
        #                     optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1
        #
        #                     # Calculate energy charged through DAM
        #                     dam_energy_charged = dam_charge_kw * 0.25  # kWh
        #                     energy_charged += dam_energy_charged
        #
        #                 # Update the vehicle's charging schedule
        #                 if idm_charge_kw > 0 and dam_charge_kw > 0:
        #                     # Mixed charging from both sources
        #                     vehicle['charging_schedule'][time] = {
        #                         'source': 'MIXED',
        #                         'kw_dam': dam_charge_kw,
        #                         'kw_idm': idm_charge_kw,
        #                         'idm_type': '15min' if use_15_min_idm else '60min'
        #                     }
        #                 elif idm_charge_kw > 0:
        #                     # Only IDM charging
        #                     vehicle['charging_schedule'][time] = {
        #                         'source': 'IDM',
        #                         'kw': idm_charge_kw,
        #                         'idm_type': '15min' if use_15_min_idm else '60min'
        #                     }
        #                 elif dam_charge_kw > 0:
        #                     # Only DAM charging
        #                     vehicle['charging_schedule'][time] = {
        #                         'source': 'DAM',
        #                         'kw': dam_charge_kw
        #                     }
        #
        #                 # Update vehicle's SOC
        #                 if energy_charged > 0:
        #                     vehicle['current_soc'] += energy_charged / vehicle['capacity_kwh']
        #             # # Výpočet, koľko vozidiel môže použiť IDM v tomto čase
        #             # vehicles_needing_charge = [v for v in eligible_vehicles
        #             #                            if v['current_soc'] < v['target_soc'] and
        #             #                            time >= v['arrival_time'] and
        #             #                            time < v['departure_time']]
        #             #
        #             # # Zoradenie vozidiel podľa potreby nabíjania (zostupne - priorita pre vozidlá, ktoré potrebujú viac nabíjania)
        #             # vehicles_needing_charge.sort(key=lambda v: v['target_soc'] - v['current_soc'], reverse=True)
        #             #
        #             # # Alokácia IDM nabíjania
        #             # idm_allocated_kw = 0
        #             #
        #             # for vehicle in vehicles_needing_charge:
        #             #     if idm_allocated_kw >= idm_potential_kw:
        #             #         break
        #             #
        #             #     # Výpočet potrebnej energie
        #             #     remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle[
        #             #         'capacity_kwh']
        #             #
        #             #     if remaining_energy_needed <= 0:
        #             #         continue
        #             #
        #             #     # Výpočet, koľko možno nabiť v tejto 15-minútovej perióde
        #             #     max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minút = 0.25 hodiny
        #             #     charge_amount = min(max_energy_per_period, remaining_energy_needed)
        #             #     charge_kw = charge_amount / 0.25  # Konverzia späť na kW
        #             #
        #             #     # Zabezpečenie, aby sme neprekročili IDM krokový limit
        #             #     if idm_allocated_kw + charge_kw > idm_potential_kw:
        #             #         charge_kw = idm_potential_kw - idm_allocated_kw
        #             #         charge_amount = charge_kw * 0.25
        #             #
        #             #     if charge_kw > 0:
        #             #         # Aktualizácia harmonogramu na základe toho, ktorý IDM trh používame
        #             #         if use_15_min_idm:
        #             #             optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] += charge_kw
        #             #             optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] += 1
        #             #             vehicles_idm_15_charged += 1
        #             #         else:
        #             #             optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] += charge_kw
        #             #             optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] += 1
        #             #             vehicles_idm_60_charged += 1
        #             #
        #             #         # Aktualizácia celkového IDM a celkového nabíjania
        #             #         optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] += charge_kw
        #             #         optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += charge_kw
        #             #         optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] += 1
        #             #
        #             #         # Aktualizácia nabíjacieho plánu a SOC vozidla
        #             #         if time in vehicle['charging_schedule']:
        #             #             vehicle['charging_schedule'][time] = {
        #             #                 'source': 'MIXED',
        #             #                 'kw_dam': vehicle['charging_schedule'][time]['kw'],
        #             #                 'kw_idm': charge_kw,
        #             #                 'idm_type': '15min' if use_15_min_idm else '60min'
        #             #             }
        #             #         else:
        #             #             vehicle['charging_schedule'][time] = {
        #             #                 'source': 'IDM',
        #             #                 'kw': charge_kw,
        #             #                 'idm_type': '15min' if use_15_min_idm else '60min'
        #             #             }
        #             #
        #             #         # Aktualizácia aktuálneho SOC vozidla
        #             #         vehicle['current_soc'] += charge_amount / vehicle['capacity_kwh']
        #             #
        #             #         # Aktualizácia alokovaného IDM výkonu
        #             #         idm_allocated_kw += charge_kw

        # Spracovanie každého časového slotu pre IDM nabíjanie
        for time in time_slots:
            # Získanie riadku pre tento čas
            row = optimized.loc[optimized['Time'] == time].iloc[0]

            # Find all vehicles needing charge at this time
            vehicles_needing_charge = [v for v in eligible_vehicles
                                       if v['current_soc'] < v['target_soc'] and
                                       time >= v['arrival_time'] and
                                       time < v['departure_time']]

            # Skip if no vehicles need charging at this time
            if not vehicles_needing_charge:
                continue

            # Determine which IDM market to use (15-min or 60-min) based on price
            use_15_min_idm = row['idm_15_price'] <= row['idm_60_price']
            idm_price = row['idm_15_price'] if use_15_min_idm else row['idm_60_price']

            # Process each vehicle needing charge
            for vehicle in vehicles_needing_charge:
                # Calculate energy needed
                remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle['capacity_kwh']

                if remaining_energy_needed <= 0:
                    continue

                # Calculate how much power we can charge in this 15-minute period
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minutes = 0.25 hour
                charge_amount = min(max_energy_per_period, remaining_energy_needed)
                charge_kw = charge_amount / 0.25  # Convert back to kW

                # DIRECT APPROACH: Always use IDM for as much as possible in 100kW steps
                # Calculate IDM portion (in 100kW steps)
                idm_charge_kw = int(charge_kw / 100) * 100
                print(idm_charge_kw)
                # Calculate DAM portion (remainder)
                dam_charge_kw = charge_kw - idm_charge_kw

                # Update charging information
                energy_charged = 0

                # Process IDM portion if any
                if idm_charge_kw > 0:
                    # Update the schedule based on which IDM market we're using
                    if use_15_min_idm:
                        optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] += 1
                        vehicles_idm_15_charged += 1
                    else:
                        optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] += 1
                        vehicles_idm_60_charged += 1

                    # Update total IDM and total charging
                    optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] += 1

                    # Calculate energy charged through IDM
                    idm_energy_charged = idm_charge_kw * 0.25  # kWh
                    energy_charged += idm_energy_charged

                # Process DAM portion if any
                if dam_charge_kw > 0:
                    # Update DAM charges
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Calculate energy charged through DAM
                    dam_energy_charged = dam_charge_kw * 0.25  # kWh
                    energy_charged += dam_energy_charged

                # Update the vehicle's charging schedule
                if idm_charge_kw > 0 and dam_charge_kw > 0:
                    # Mixed charging from both sources
                    vehicle['charging_schedule'][time] = {
                        'source': 'MIXED',
                        'kw_dam': dam_charge_kw,
                        'kw_idm': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif idm_charge_kw > 0:
                    # Only IDM charging
                    vehicle['charging_schedule'][time] = {
                        'source': 'IDM',
                        'kw': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif dam_charge_kw > 0:
                    # Only DAM charging
                    vehicle['charging_schedule'][time] = {
                        'source': 'DAM',
                        'kw': dam_charge_kw
                    }

                # Update vehicle's SOC
                if energy_charged > 0:
                    vehicle['current_soc'] += energy_charged / vehicle['capacity_kwh']


        # Aktualizácia celkového počtu nabíjajúcich sa vozidiel v každom časovom kroku
        optimized['total_ev_charging'] = (optimized['total_ev_charging_dam'] +
                                          optimized['total_ev_charging_idm_15'] +
                                          optimized['total_ev_charging_idm_60'])

        # Výpočet nákladov
        optimized['dam_cost'] = optimized['dam_charge_kw'] * optimized[
            'dam_price'] / 1000 * 0.25  # Konverzia na MWh náklady pre 15 min
        optimized['idm_15_cost'] = optimized['idm_15_charge_kw'] * optimized['idm_15_price'] / 1000 * 0.25
        optimized['idm_60_cost'] = optimized['idm_60_charge_kw'] * optimized['idm_60_price'] / 1000 * 0.25
        optimized['idm_cost'] = optimized['idm_15_cost'] + optimized['idm_60_cost']

        # Výpočet celkovej energie a nákladov
        total_dam_energy = optimized['dam_charge_kw'].sum() * 0.25  # kWh
        total_idm_15_energy = optimized['idm_15_charge_kw'].sum() * 0.25  # kWh
        total_idm_60_energy = optimized['idm_60_charge_kw'].sum() * 0.25  # kWh
        total_energy = total_dam_energy + total_idm_15_energy + total_idm_60_energy

        total_dam_cost = optimized['dam_cost'].sum()
        total_idm_15_cost = optimized['idm_15_cost'].sum()
        total_idm_60_cost = optimized['idm_60_cost'].sum()
        total_cost = total_dam_cost + total_idm_15_cost + total_idm_60_cost

        # Výpočet, koľko by sme zaplatili s použitím iba DAM cien
        hypothetical_dam_cost = (
                total_dam_energy * optimized['dam_price'].mean() / 1000 +
                total_idm_15_energy * optimized['dam_price'].mean() / 1000 +
                total_idm_60_energy * optimized['dam_price'].mean() / 1000
        )

        # Výpočet pôvodných IDM nákladov (bez zliav)
        original_idm_15_cost = optimized['idm_15_charge_kw'] * optimized['original_idm_15_price'] / 1000 * 0.25
        original_idm_60_cost = optimized['idm_60_charge_kw'] * optimized['original_idm_60_price'] / 1000 * 0.25
        original_idm_cost = original_idm_15_cost.sum() + original_idm_60_cost.sum()
        original_total_cost = total_dam_cost + original_idm_cost

        print(f"\nSúhrn optimalizovaného nabíjania pre {simulation_days} dní:")
        print(f"Celkový počet nabíjaných vozidiel: {vehicles_dam_charged} (DAM), "
              f"{vehicles_idm_15_charged} (IDM 15-min), {vehicles_idm_60_charged} (IDM 60-min)")
        print(f"Celková nabitá energia: {total_energy:.2f} kWh")
        print(f"- DAM: {total_dam_energy:.2f} kWh ({(total_dam_energy / total_energy * 100):.1f}%)")
        print(f"- IDM 15-min: {total_idm_15_energy:.2f} kWh ({(total_idm_15_energy / total_energy * 100):.1f}%)")
        print(f"- IDM 60-min: {total_idm_60_energy:.2f} kWh ({(total_idm_60_energy / total_energy * 100):.1f}%)")

        print(f"\nNáklady s aplikovanými optimalizačnými možnosťami:")
        print(f"Celkové náklady: {total_cost:.2f} €")
        print(f"- DAM: {total_dam_cost:.2f} € ({(total_dam_cost / total_cost * 100):.1f}%)")
        print(f"- IDM 15-min: {total_idm_15_cost:.2f} € ({(total_idm_15_cost / total_cost * 100):.1f}%)")
        print(f"- IDM 60-min: {total_idm_60_cost:.2f} € ({(total_idm_60_cost / total_cost * 100):.1f}%)")
        print(f"Priemerná cena: {(total_cost / total_energy * 1000):.2f} €/MWh")

        if general_discount < 1.0 or night_discount < 1.0:
            print(f"\nNáklady bez aplikovaných zliav (pôvodné ceny):")
            print(f"Celkové náklady: {original_total_cost:.2f} €")
            print(f"- DAM: {total_dam_cost:.2f} € ({(total_dam_cost / original_total_cost * 100):.1f}%)")
            print(
                f"- IDM (pôvodné ceny): {original_idm_cost:.2f} € ({(original_idm_cost / original_total_cost * 100):.1f}%)")
            print(f"Priemerná cena: {(original_total_cost / total_energy * 1000):.2f} €/MWh")

        print(f"\nHypotetické náklady iba s DAM: {hypothetical_dam_cost:.2f} €")
        print(f"Úspory oproti čistému DAM: {(hypothetical_dam_cost - total_cost):.2f} € "
              f"({((hypothetical_dam_cost - total_cost) / hypothetical_dam_cost * 100):.1f}%)")

        self.optimized_results = optimized
        return optimized

    def optimized_strategy_2(self, simulation_start, simulation_end=None, options=None, DAM_ALLOCATION=0.0):
        """
        Implementácia optimalizovanej stratégie nabíjania (DAM + IDM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.
        options : dict, voliteľné
            Slovník optimalizačných možností:
            - 'analyze_price_patterns': bool - Analýza hodinových cenových vzorov
            - 'general_idm_discount': float - Aplikácia všeobecnej zľavy na IDM ceny
            - 'targeted_night_discount': float - Aplikácia cielenej zľavy počas nočných hodín
        DAM_ALLOCATION : float, voliteľné
            Percento energie alokovanej na DAM (predvolene 0.4 alebo 40%)

        Výstup:
        --------
        DataFrame s optimalizovaným nabíjacím harmonogramom a nákladmi
        """
        # Nastavenie predvolených možností, ak nie sú poskytnuté
        if options is None:
            options = {
                'analyze_price_patterns': False,
                'general_idm_discount': 1,
                'targeted_night_discount': 1
            }

        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        simulation_days = (simulation_end - simulation_start).total_seconds() / (24 * 3600)
        simulation_days = max(1, round(simulation_days))

        print(f"Spúšťanie optimalizovanej stratégie od {simulation_start} do {simulation_end} ({simulation_days} dní)")
        print(f"Optimalizačné možnosti: {options}")
        print(f"DAM alokácia: {DAM_ALLOCATION * 100}%")

        # Debug information for IDM charging
        print("\n=== IDM CHARGING CONFIGURATION ===")
        print(f"Fleet size: {self.fleet_size} vehicles")
        print(f"Max charge per vehicle: {self.max_charge_kw} kW")
        print(f"Total theoretical fleet capacity: {self.fleet_size * self.max_charge_kw} kW")
        print(f"IDM minimum power step: 100 kW (0.1 MW)")
        print("=" * 40)

        # Výpočet počtu 15-minútových periód v simulácii
        simulation_periods = int((simulation_end - simulation_start).total_seconds() / 900)  # 900 sekúnd = 15 minút
        if simulation_periods <= 0:
            simulation_periods = 96  # Predvolene 24 hodín (96 periód po 15 minút)

        print(f"Vytváranie harmonogramu s {simulation_periods} 15-minútovými periódami")

        # Vytvorenie harmonogramu s 15-minútovými intervalmi pre celé simulačné obdobie
        schedule_times = pd.date_range(
            start=simulation_start,
            periods=simulation_periods,
            freq='15T'
        )

        optimized = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_charge_kw': 0.0,
            'idm_charge_kw': 0.0,
            'idm_15_charge_kw': 0.0,
            'idm_60_charge_kw': 0.0,
            'dam_price': 0.0,
            'idm_price': 0.0,
            'idm_15_price': 0.0,
            'idm_60_price': 0.0,
            'original_idm_15_price': 0.0,
            'original_idm_60_price': 0.0,
            'dam_cost': 0.0,
            'idm_cost': 0.0,
            'idm_15_cost': 0.0,
            'idm_60_cost': 0.0,
            'total_ev_charging': 0,
            'total_ev_charging_dam': 0,
            'total_ev_charging_idm': 0,
            'total_ev_charging_idm_15': 0,
            'total_ev_charging_idm_60': 0
        })

        # Získanie vhodných vozidiel pre simulačné obdobie
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"{len(eligible_vehicles)} vozidiel je vhodných na nabíjanie v tomto období")

        # Získanie cenových dát
        dam_period_prices = self.dam_prices.copy()
        idm_15_period_prices = self.idm_15_prices.copy()
        idm_60_period_prices = self.idm_60_prices.copy()

        # Mapovanie cien na harmonogram
        for idx, row in optimized.iterrows():
            time = row['Time']
            hour_start = time.replace(minute=0, second=0, microsecond=0)

            # Nájdenie DAM ceny (najbližšia hodinová zhoda)
            dam_prices_in_range = dam_period_prices[
                (dam_period_prices['Time'] >= hour_start) &
                (dam_period_prices['Time'] < hour_start + timedelta(hours=1))
                ]

            if not dam_prices_in_range.empty:
                optimized.at[idx, 'dam_price'] = dam_prices_in_range.iloc[0]['cena']
            else:
                closest_idx = (dam_period_prices['Time'] - hour_start).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(dam_period_prices):
                    optimized.at[idx, 'dam_price'] = dam_period_prices.iloc[closest_idx]['cena']
                else:
                    optimized.at[idx, 'dam_price'] = 50.0  # Predvolená hodnota

            # Nájdenie IDM 15-min ceny
            idm_15_prices_in_range = idm_15_period_prices[
                (idm_15_period_prices['Time'] >= time) &
                (idm_15_period_prices['Time'] < time + timedelta(minutes=15))
                ]

            if not idm_15_prices_in_range.empty:
                optimized.at[idx, 'idm_15_price'] = idm_15_prices_in_range.iloc[0]['cena']
                optimized.at[idx, 'original_idm_15_price'] = idm_15_prices_in_range.iloc[0]['cena']
            else:
                closest_idx = (idm_15_period_prices['Time'] - time).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(idm_15_period_prices):
                    optimized.at[idx, 'idm_15_price'] = idm_15_period_prices.iloc[closest_idx]['cena']
                    optimized.at[idx, 'original_idm_15_price'] = idm_15_period_prices.iloc[closest_idx]['cena']
                else:
                    optimized.at[idx, 'idm_15_price'] = optimized.at[idx, 'dam_price']
                    optimized.at[idx, 'original_idm_15_price'] = optimized.at[idx, 'dam_price']

            # Nájdenie IDM 60-min ceny
            idm_60_prices_in_range = idm_60_period_prices[
                (idm_60_period_prices['Time'] >= hour_start) &
                (idm_60_period_prices['Time'] < hour_start + timedelta(hours=1))
                ]

            if not idm_60_prices_in_range.empty:
                optimized.at[idx, 'idm_60_price'] = idm_60_prices_in_range.iloc[0]['cena']
                optimized.at[idx, 'original_idm_60_price'] = idm_60_prices_in_range.iloc[0]['cena']
            else:
                closest_idx = (idm_60_period_prices['Time'] - hour_start).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(idm_60_period_prices):
                    optimized.at[idx, 'idm_60_price'] = idm_60_period_prices.iloc[closest_idx]['cena']
                    optimized.at[idx, 'original_idm_60_price'] = idm_60_period_prices.iloc[closest_idx]['cena']
                else:
                    optimized.at[idx, 'idm_60_price'] = optimized.at[idx, 'dam_price']
                    optimized.at[idx, 'original_idm_60_price'] = optimized.at[idx, 'dam_price']

        # Aplikácia všeobecnej zľavy na IDM ceny, ak je požadovaná
        general_discount = options.get('general_idm_discount', 1.0)
        if general_discount < 1.0:
            print(f"Aplikácia všeobecného diskontného faktora IDM: {general_discount:.2f}")
            optimized['idm_15_price'] = optimized['idm_15_price'] * general_discount
            optimized['idm_60_price'] = optimized['idm_60_price'] * general_discount

        # Aplikácia cielenej zľavy počas nočných hodín, ak je požadovaná
        night_discount = options.get('targeted_night_discount', 1.0)
        if night_discount < 1.0:
            print(f"Aplikácia cieleného nočného diskontného faktora: {night_discount:.2f}")
            night_mask = (optimized['Time'].dt.hour >= 22) | (optimized['Time'].dt.hour < 6)
            optimized.loc[night_mask, 'idm_15_price'] = optimized.loc[night_mask, 'idm_15_price'] * night_discount
            optimized.loc[night_mask, 'idm_60_price'] = optimized.loc[night_mask, 'idm_60_price'] * night_discount

        # Výber lepšej IDM ceny (minimum z 15-min a 60-min)
        optimized['idm_price'] = optimized[['idm_15_price', 'idm_60_price']].min(axis=1)

        # Počítadlá pre nabíjané vozidlá
        vehicles_dam_charged = 0
        vehicles_idm_15_charged = 0
        vehicles_idm_60_charged = 0

        # KROK 1: Alokácia DAM nabíjania pre základnú časť
        print("\nAlokácia DAM nabíjania...")

        for vehicle in eligible_vehicles:
            # Výpočet potrebnej energie
            energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']

            if energy_needed_kwh <= 0:
                continue  # Vozidlo nepotrebuje nabíjanie

            # Nájdenie nabíjacieho okna
            charging_start = max(vehicle['arrival_time'], optimized['Time'].min())
            charging_end = min(vehicle['departure_time'], optimized['Time'].max())

            # Získanie všetkých časových slotov v nabíjacom okne
            charging_slots = optimized[
                (optimized['Time'] >= charging_start) &
                (optimized['Time'] < charging_end)
                ]

            if len(charging_slots) == 0:
                print(f"Varovanie: Žiadne platné nabíjacie sloty pre vozidlo {vehicle['id']}")
                continue  # Žiadne platné nabíjacie sloty

            # Použitie fixnej DAM alokácie pre všetky sloty
            dam_allocations = [(slot['Time'], DAM_ALLOCATION) for _, slot in charging_slots.iterrows()]
            dam_allocations.sort(key=lambda x: optimized.loc[optimized['Time'] == x[0], 'dam_price'].iloc[0])

            # Výpočet celkovej energie na alokáciu do DAM
            total_dam_allocation = sum(allocation for _, allocation in dam_allocations) * energy_needed_kwh / len(
                dam_allocations)
            remaining_dam_energy = total_dam_allocation
            vehicle_schedule = {}
            periods_charged = 0

            # Alokácia DAM nabíjania
            for time, _ in dam_allocations:
                if remaining_dam_energy <= 0:
                    break

                # Výpočet maximálnej energie, ktorú možno nabiť v tejto 15-minútovej perióde
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minút = 0.25 hodiny
                charge_amount = min(max_energy_per_period, remaining_dam_energy)
                charge_kw = charge_amount / 0.25  # Konverzia späť na kW

                if charge_kw > 0:
                    # Aktualizácia harmonogramu
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging'] += 1
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Aktualizácia nabíjacieho plánu vozidla
                    vehicle_schedule[time] = {'source': 'DAM', 'kw': charge_kw}

                    # Aktualizácia zostávajúcej energie
                    remaining_dam_energy -= charge_amount
                    periods_charged += 1

            # Aktualizácia aktuálneho SOC vozidla na základe DAM nabíjania
            energy_charged_dam = total_dam_allocation - remaining_dam_energy
            vehicle['current_soc'] = vehicle['return_soc'] + (energy_charged_dam / vehicle['capacity_kwh'])

            # Uloženie DAM časti nabíjacieho plánu
            vehicle['charging_schedule'] = vehicle_schedule

            if periods_charged > 0:
                vehicles_dam_charged += 1

        print(f"DAM nabíjanie alokované pre {vehicles_dam_charged} vozidiel")

        # KROK 2: Teraz spracujeme IDM nabíjanie pre zostávajúcu energiu
        print("\nAlokácia IDM nabíjania...")

        # Debug počítadlá pre IDM
        idm_opportunities = 0
        idm_allocations_made = 0

        # Spracovanie každého vozidla pre IDM nabíjanie
        for vehicle in eligible_vehicles:
            # Kontrola, či vozidlo ešte potrebuje energiu
            if vehicle['current_soc'] >= vehicle['target_soc']:
                continue

            # Výpočet zostávajúcej potrebnej energie
            remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle['capacity_kwh']

            if remaining_energy_needed <= 0:
                continue

            # Nájdenie nabíjacieho okna
            charging_start = max(vehicle['arrival_time'], optimized['Time'].min())
            charging_end = min(vehicle['departure_time'], optimized['Time'].max())

            # Získanie všetkých časových slotov v nabíjacom okne
            charging_slots = optimized[
                (optimized['Time'] >= charging_start) &
                (optimized['Time'] < charging_end)
                ]

            if len(charging_slots) == 0:
                continue  # Žiadne platné nabíjacie sloty

            # Zoradenie podľa ceny (najnižšia IDM cena najskôr)
            charging_slots = charging_slots.sort_values('idm_price').copy()

            for _, slot in charging_slots.iterrows():
                time = slot['Time']

                # Ak je vozidlo už plne nabité, prejsť na ďalšie
                if vehicle['current_soc'] >= vehicle['target_soc']:
                    break

                # Aktualizácia zostávajúcej potrebnej energie
                remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle['capacity_kwh']

                if remaining_energy_needed <= 0:
                    break

                # Výpočet maximálnej energie, ktorú možno nabiť v tejto 15-minútovej perióde
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 minút = 0.25 hodiny
                charge_amount = min(max_energy_per_period, remaining_energy_needed)
                charge_kw = charge_amount / 0.25  # Konverzia späť na kW

                # Určenie, ktorý IDM trh použiť (15-min alebo 60-min)
                use_15_min_idm = slot['idm_15_price'] <= slot['idm_60_price']

                # KĽÚČOVÁ ČASŤ: Rozdelenie nabíjania na IDM a DAM
                # IDM musí byť v 100kW krokoch, zvyšok ide na DAM

                # Zaokrúhlenie na celé 100kW pre IDM
                idm_charge_kw = int(charge_kw / 100) * 100

                # Zvyšok ide na DAM
                dam_charge_kw = charge_kw - idm_charge_kw

                # Debug výpis
                if charge_kw >= 100:
                    idm_opportunities += 1
                    print(
                        f"Vozidlo {vehicle['id']} v čase {time}: potrebuje {charge_kw:.2f} kW -> IDM: {idm_charge_kw} kW, DAM: {dam_charge_kw:.2f} kW")

                # Spracovanie IDM časti (ak je aspoň 100kW)
                if idm_charge_kw >= 100:
                    idm_allocations_made += 1

                    # Aktualizácia údajov podľa typu IDM
                    if use_15_min_idm:
                        optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] += 1
                        vehicles_idm_15_charged += 1
                    else:
                        optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] += 1
                        vehicles_idm_60_charged += 1

                    # Aktualizácia celkového IDM nabíjania
                    optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] += 1

                    # Aktualizácia energie nabitej pomocou IDM
                    idm_energy = idm_charge_kw * 0.25  # kWh
                    vehicle['current_soc'] += idm_energy / vehicle['capacity_kwh']

                # Spracovanie DAM časti (zvyšok)
                if dam_charge_kw > 0:
                    # Aktualizácia DAM nabíjania
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Aktualizácia energie nabitej pomocou DAM
                    dam_energy = dam_charge_kw * 0.25  # kWh
                    vehicle['current_soc'] += dam_energy / vehicle['capacity_kwh']

                # Aktualizácia nabíjacieho plánu vozidla
                if idm_charge_kw > 0 and dam_charge_kw > 0:
                    # Zmiešané nabíjanie
                    vehicle['charging_schedule'][time] = {
                        'source': 'MIXED',
                        'kw_dam': dam_charge_kw,
                        'kw_idm': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif idm_charge_kw > 0:
                    # Len IDM nabíjanie
                    vehicle['charging_schedule'][time] = {
                        'source': 'IDM',
                        'kw': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif dam_charge_kw > 0:
                    # Len DAM nabíjanie
                    if time in vehicle['charging_schedule']:
                        # Ak už existuje DAM záznam, aktualizujeme ho
                        existing_kw = vehicle['charging_schedule'][time]['kw']
                        vehicle['charging_schedule'][time] = {
                            'source': 'DAM',
                            'kw': existing_kw + dam_charge_kw
                        }
                    else:
                        # Nový DAM záznam
                        vehicle['charging_schedule'][time] = {
                            'source': 'DAM',
                            'kw': dam_charge_kw
                        }

        # Debug výpis pre IDM nabíjanie
        print(f"\nIDM príležitosti (vozidlá s potrebou >= 100kW): {idm_opportunities}")
        print(f"IDM alokácie uskutočnené: {idm_allocations_made}")
        print(f"Vozidlá nabíjané cez IDM 15-min: {vehicles_idm_15_charged}")
        print(f"Vozidlá nabíjané cez IDM 60-min: {vehicles_idm_60_charged}")

        # Aktualizácia celkového počtu nabíjajúcich sa vozidiel v každom časovom kroku
        optimized['total_ev_charging'] = (optimized['total_ev_charging_dam'] +
                                          optimized['total_ev_charging_idm_15'] +
                                          optimized['total_ev_charging_idm_60'])

        # Výpočet nákladov
        optimized['dam_cost'] = optimized['dam_charge_kw'] * optimized[
            'dam_price'] / 1000 * 0.25  # Konverzia na MWh náklady pre 15 min
        optimized['idm_15_cost'] = optimized['idm_15_charge_kw'] * optimized['idm_15_price'] / 1000 * 0.25
        optimized['idm_60_cost'] = optimized['idm_60_charge_kw'] * optimized['idm_60_price'] / 1000 * 0.25
        optimized['idm_cost'] = optimized['idm_15_cost'] + optimized['idm_60_cost']

        # Výpočet celkovej energie a nákladov
        total_dam_energy = optimized['dam_charge_kw'].sum() * 0.25  # kWh
        total_idm_15_energy = optimized['idm_15_charge_kw'].sum() * 0.25  # kWh
        total_idm_60_energy = optimized['idm_60_charge_kw'].sum() * 0.25  # kWh
        total_energy = total_dam_energy + total_idm_15_energy + total_idm_60_energy

        total_dam_cost = optimized['dam_cost'].sum()
        total_idm_15_cost = optimized['idm_15_cost'].sum()
        total_idm_60_cost = optimized['idm_60_cost'].sum()
        total_cost = total_dam_cost + total_idm_15_cost + total_idm_60_cost

        # Výpis súhrnu
        print(f"\nSúhrn optimalizovaného nabíjania:")
        print(f"Celková nabitá energia: {total_energy:.2f} kWh")
        print(f"- DAM: {total_dam_energy:.2f} kWh ({(total_dam_energy / total_energy * 100):.1f}%)")
        print(f"- IDM 15-min: {total_idm_15_energy:.2f} kWh ({(total_idm_15_energy / total_energy * 100):.1f}%)")
        print(f"- IDM 60-min: {total_idm_60_energy:.2f} kWh ({(total_idm_60_energy / total_energy * 100):.1f}%)")

        print(f"\nCelkové náklady: {total_cost:.2f} €")
        print(f"- DAM: {total_dam_cost:.2f} € ({(total_dam_cost / total_cost * 100):.1f}%)")
        print(f"- IDM 15-min: {total_idm_15_cost:.2f} € ({(total_idm_15_cost / total_cost * 100):.1f}%)")
        print(f"- IDM 60-min: {total_idm_60_cost:.2f} € ({(total_idm_60_cost / total_cost * 100):.1f}%)")
        print(f"Priemerná cena: {(total_cost / total_energy * 1000):.2f} €/MWh")

        # Uloženie výsledkov
        self.optimized_results = optimized
        return optimized

    def optimized_strategy_3(self, simulation_start, simulation_end=None, options=None, DAM_ALLOCATION=0.05):
        """
        Implementácia optimalizovanej stratégie nabíjania (DAM + IDM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.
        options : dict, voliteľné
            Slovník optimalizačných možností (unused in simplified version)
        DAM_ALLOCATION : float, voliteľné
            Percento energie alokovanej na DAM (predvolene 0.05 alebo 5%)

        Výstup:
        --------
        DataFrame s optimalizovaným nabíjacím harmonogramom a nákladmi
        """
        print("Starting simplified optimized strategy with forced IDM usage")

        # Basic setup
        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        # Create time schedule with 15-minute intervals
        schedule_times = pd.date_range(
            start=simulation_start,
            end=simulation_end,
            freq='15T'
        )

        # Initialize results dataframe
        optimized = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_charge_kw': 0.0,
            'idm_charge_kw': 0.0,
            'idm_15_charge_kw': 0.0,
            'idm_60_charge_kw': 0.0,
            'dam_price': 0.0,
            'idm_price': 0.0,
            'idm_15_price': 0.0,
            'idm_60_price': 0.0,
            'dam_cost': 0.0,
            'idm_cost': 0.0,
            'idm_15_cost': 0.0,
            'idm_60_cost': 0.0,
            'total_ev_charging': 0,
            'total_ev_charging_dam': 0,
            'total_ev_charging_idm': 0,
            'total_ev_charging_idm_15': 0,
            'total_ev_charging_idm_60': 0
        })

        # Get eligible vehicles
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"Found {len(eligible_vehicles)} eligible vehicles")

        # Load price data
        # Get DAM and IDM prices and assign to time slots
        for idx, row in optimized.iterrows():
            time = row['Time']
            hour = time.replace(minute=0, second=0, microsecond=0)

            # Find DAM price (closest hourly match)
            dam_price_match = self.dam_prices[
                (self.dam_prices['Time'] >= hour) &
                (self.dam_prices['Time'] < hour + timedelta(hours=1))
                ]
            if not dam_price_match.empty:
                optimized.at[idx, 'dam_price'] = dam_price_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'dam_price'] = 50.0

            # Find IDM 15-min price
            idm_15_match = self.idm_15_prices[
                (self.idm_15_prices['Time'] >= time) &
                (self.idm_15_prices['Time'] < time + timedelta(minutes=15))
                ]
            if not idm_15_match.empty:
                optimized.at[idx, 'idm_15_price'] = idm_15_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'idm_15_price'] = optimized.at[idx, 'dam_price']

            # Find IDM 60-min price
            idm_60_match = self.idm_60_prices[
                (self.idm_60_prices['Time'] >= hour) &
                (self.idm_60_prices['Time'] < hour + timedelta(hours=1))
                ]
            if not idm_60_match.empty:
                optimized.at[idx, 'idm_60_price'] = idm_60_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'idm_60_price'] = optimized.at[idx, 'dam_price']

            # Set IDM price to the better of 15-min and 60-min
            optimized.at[idx, 'idm_price'] = min(
                optimized.at[idx, 'idm_15_price'],
                optimized.at[idx, 'idm_60_price']
            )

        # Counters for charging vehicles
        vehicles_dam_charged = 0
        vehicles_idm_15_charged = 0
        vehicles_idm_60_charged = 0

        print("Processing each vehicle...")
        # Process each vehicle
        for vehicle_idx, vehicle in enumerate(eligible_vehicles):
            print(f"Processing vehicle {vehicle_idx + 1}/{len(eligible_vehicles)}: {vehicle['id']}")

            # Calculate energy needed
            energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']
            if energy_needed_kwh <= 0:
                print(f"  Vehicle {vehicle['id']} doesn't need charging")
                continue

            # Get all valid charging slots for this vehicle
            valid_slots = optimized[
                (optimized['Time'] >= vehicle['arrival_time']) &
                (optimized['Time'] < vehicle['departure_time'])
                ]
            if len(valid_slots) == 0:
                print(f"  No valid charging slots for vehicle {vehicle['id']}")
                continue

            # Reset current SOC to return SOC
            vehicle['current_soc'] = vehicle['return_soc']
            vehicle['charging_schedule'] = {}

            # Sort charging slots by price
            valid_slots = valid_slots.sort_values('dam_price')

            # STEP 1: Allocate some energy to DAM (fixed percentage)
            dam_energy_to_allocate = energy_needed_kwh * DAM_ALLOCATION
            remaining_dam_energy = dam_energy_to_allocate

            for _, slot in valid_slots.iterrows():
                if remaining_dam_energy <= 0:
                    break

                time = slot['Time']

                # How much can we charge in this 15-min period
                max_charge_per_period = vehicle['max_charge_kw'] * 0.25  # 15 min = 0.25 hour
                charge_amount = min(max_charge_per_period, remaining_dam_energy)
                dam_charge_kw = charge_amount / 0.25  # Convert back to kW

                if dam_charge_kw > 0:
                    # Update dataframe
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging'] += 1
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Update vehicle
                    vehicle['charging_schedule'][time] = {'source': 'DAM', 'kw': dam_charge_kw}
                    vehicle['current_soc'] += charge_amount / vehicle['capacity_kwh']

                    # Update remaining energy
                    remaining_dam_energy -= charge_amount

            if dam_energy_to_allocate - remaining_dam_energy > 0:
                vehicles_dam_charged += 1
                print(f"  Allocated {dam_energy_to_allocate - remaining_dam_energy:.2f} kWh to DAM")

            # STEP 2: Allocate remaining energy to IDM (in 100kW steps) and DAM (remainder)
            # Recalculate remaining energy needed
            remaining_energy_needed = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle['capacity_kwh']
            if remaining_energy_needed <= 0:
                continue

            print(f"  Vehicle {vehicle['id']} still needs {remaining_energy_needed:.2f} kWh")

            # Sort slots by IDM price for IDM allocation
            valid_slots = optimized[
                (optimized['Time'] >= vehicle['arrival_time']) &
                (optimized['Time'] < vehicle['departure_time'])
                ].sort_values('idm_price')

            for _, slot in valid_slots.iterrows():
                if remaining_energy_needed <= 0:
                    break

                time = slot['Time']

                # How much can we charge in this 15-min period
                max_charge_per_period = vehicle['max_charge_kw'] * 0.25  # 15 min = 0.25 hour
                charge_amount = min(max_charge_per_period, remaining_energy_needed)
                total_charge_kw = charge_amount / 0.25  # Convert back to kW

                # CRITICAL PART: Determine IDM and DAM allocation
                # IDM must be in 100kW steps, remainder goes to DAM
                idm_charge_kw = int(total_charge_kw / 100) * 100
                dam_charge_kw = total_charge_kw - idm_charge_kw

                print(
                    f"  TIME {time}: Total charge needed: {total_charge_kw:.2f} kW -> IDM: {idm_charge_kw} kW, DAM: {dam_charge_kw:.2f} kW")

                # Determine which IDM market to use
                use_15_min_idm = slot['idm_15_price'] <= slot['idm_60_price']

                # Process IDM charging if any
                if idm_charge_kw > 0:
                    # Update appropriate IDM counter
                    if use_15_min_idm:
                        optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] += 1
                        if vehicle['id'] not in [v for v in range(vehicles_idm_15_charged)]:
                            vehicles_idm_15_charged += 1
                    else:
                        optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] += idm_charge_kw
                        optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] += 1
                        if vehicle['id'] not in [v for v in range(vehicles_idm_60_charged)]:
                            vehicles_idm_60_charged += 1

                    # Update total IDM
                    optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += idm_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] += 1

                    # Calculate energy charged via IDM
                    idm_energy = idm_charge_kw * 0.25  # kWh
                    vehicle['current_soc'] += idm_energy / vehicle['capacity_kwh']
                    remaining_energy_needed -= idm_energy

                    print(f"    IDM energy charged: {idm_energy:.2f} kWh")

                # Process DAM charging if any
                if dam_charge_kw > 0:
                    optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_charge_kw'] += dam_charge_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] += 1

                    # Calculate energy charged via DAM
                    dam_energy = dam_charge_kw * 0.25  # kWh
                    vehicle['current_soc'] += dam_energy / vehicle['capacity_kwh']
                    remaining_energy_needed -= dam_energy

                    print(f"    DAM energy charged: {dam_energy:.2f} kWh")

                # Update vehicle's charging schedule
                if idm_charge_kw > 0 and dam_charge_kw > 0:
                    # Mixed charging
                    vehicle['charging_schedule'][time] = {
                        'source': 'MIXED',
                        'kw_dam': dam_charge_kw,
                        'kw_idm': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif idm_charge_kw > 0:
                    # Only IDM
                    vehicle['charging_schedule'][time] = {
                        'source': 'IDM',
                        'kw': idm_charge_kw,
                        'idm_type': '15min' if use_15_min_idm else '60min'
                    }
                elif dam_charge_kw > 0:
                    # Only DAM
                    if time in vehicle['charging_schedule']:
                        # Update existing DAM entry
                        if vehicle['charging_schedule'][time]['source'] == 'DAM':
                            vehicle['charging_schedule'][time]['kw'] += dam_charge_kw
                        else:
                            # This shouldn't happen with this logic, but just in case
                            vehicle['charging_schedule'][time] = {
                                'source': 'DAM',
                                'kw': dam_charge_kw
                            }
                    else:
                        # New DAM entry
                        vehicle['charging_schedule'][time] = {
                            'source': 'DAM',
                            'kw': dam_charge_kw
                        }

        # Calculate costs
        optimized['dam_cost'] = optimized['dam_charge_kw'] * optimized['dam_price'] / 1000 * 0.25
        optimized['idm_15_cost'] = optimized['idm_15_charge_kw'] * optimized['idm_15_price'] / 1000 * 0.25
        optimized['idm_60_cost'] = optimized['idm_60_charge_kw'] * optimized['idm_60_price'] / 1000 * 0.25
        optimized['idm_cost'] = optimized['idm_15_cost'] + optimized['idm_60_cost']

        # Calculate total energy and costs
        total_dam_energy = optimized['dam_charge_kw'].sum() * 0.25
        total_idm_15_energy = optimized['idm_15_charge_kw'].sum() * 0.25
        total_idm_60_energy = optimized['idm_60_charge_kw'].sum() * 0.25
        total_energy = total_dam_energy + total_idm_15_energy + total_idm_60_energy

        total_dam_cost = optimized['dam_cost'].sum()
        total_idm_15_cost = optimized['idm_15_cost'].sum()
        total_idm_60_cost = optimized['idm_60_cost'].sum()
        total_cost = total_dam_cost + total_idm_15_cost + total_idm_60_cost

        # Print summary
        print("\nOptimized Charging Summary:")
        print(f"Total energy charged: {total_energy:.2f} kWh")
        print(
            f"- DAM: {total_dam_energy:.2f} kWh ({total_dam_energy / total_energy * 100 if total_energy > 0 else 0:.1f}%)")
        print(
            f"- IDM 15-min: {total_idm_15_energy:.2f} kWh ({total_idm_15_energy / total_energy * 100 if total_energy > 0 else 0:.1f}%)")
        print(
            f"- IDM 60-min: {total_idm_60_energy:.2f} kWh ({total_idm_60_energy / total_energy * 100 if total_energy > 0 else 0:.1f}%)")

        print(f"\nTotal cost: {total_cost:.2f} €")
        print(f"- DAM: {total_dam_cost:.2f} € ({total_dam_cost / total_cost * 100 if total_cost > 0 else 0:.1f}%)")
        print(
            f"- IDM 15-min: {total_idm_15_cost:.2f} € ({total_idm_15_cost / total_cost * 100 if total_cost > 0 else 0:.1f}%)")
        print(
            f"- IDM 60-min: {total_idm_60_cost:.2f} € ({total_idm_60_cost / total_cost * 100 if total_cost > 0 else 0:.1f}%)")

        # Store results
        self.optimized_results = optimized
        return optimized

    def optimized_strategy_4_ok(self, simulation_start, simulation_end=None, options=None, DAM_ALLOCATION=0.05):
        """
        Implementácia optimalizovanej stratégie nabíjania (DAM + IDM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.
        options : dict, voliteľné
            Slovník optimalizačných možností
        DAM_ALLOCATION : float, voliteľné
            Percento energie alokovanej na DAM (predvolene 0.05 alebo 5%)
        """
        print("Starting optimized strategy with total fleet IDM step constraints")

        # Basic setup
        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        # Create time schedule with 15-minute intervals
        schedule_times = pd.date_range(
            start=simulation_start,
            end=simulation_end,
            freq='15T'
        )

        # Initialize results dataframe
        optimized = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_charge_kw': 0.0,
            'idm_charge_kw': 0.0,
            'idm_15_charge_kw': 0.0,
            'idm_60_charge_kw': 0.0,
            'dam_price': 0.0,
            'idm_price': 0.0,
            'idm_15_price': 0.0,
            'idm_60_price': 0.0,
            'dam_cost': 0.0,
            'idm_cost': 0.0,
            'idm_15_cost': 0.0,
            'idm_60_cost': 0.0,
            'total_ev_charging': 0,
            'total_ev_charging_dam': 0,
            'total_ev_charging_idm': 0,
            'total_ev_charging_idm_15': 0,
            'total_ev_charging_idm_60': 0
        })

        # Get eligible vehicles
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"Found {len(eligible_vehicles)} eligible vehicles")

        # Load price data - Get DAM and IDM prices and assign to time slots
        for idx, row in optimized.iterrows():
            time = row['Time']
            hour = time.replace(minute=0, second=0, microsecond=0)

            # Find DAM price
            dam_price_match = self.dam_prices[
                (self.dam_prices['Time'] >= hour) &
                (self.dam_prices['Time'] < hour + timedelta(hours=1))
                ]
            if not dam_price_match.empty:
                optimized.at[idx, 'dam_price'] = dam_price_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'dam_price'] = 50.0

            # Find IDM 15-min price
            idm_15_match = self.idm_15_prices[
                (self.idm_15_prices['Time'] >= time) &
                (self.idm_15_prices['Time'] < time + timedelta(minutes=15))
                ]
            if not idm_15_match.empty:
                optimized.at[idx, 'idm_15_price'] = idm_15_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'idm_15_price'] = optimized.at[idx, 'dam_price']

            # Find IDM 60-min price
            idm_60_match = self.idm_60_prices[
                (self.idm_60_prices['Time'] >= hour) &
                (self.idm_60_prices['Time'] < hour + timedelta(hours=1))
                ]
            if not idm_60_match.empty:
                optimized.at[idx, 'idm_60_price'] = idm_60_match.iloc[0]['cena']
            else:
                # Default if no match
                optimized.at[idx, 'idm_60_price'] = optimized.at[idx, 'dam_price']

            # Set IDM price to the better of 15-min and 60-min
            optimized.at[idx, 'idm_price'] = min(
                optimized.at[idx, 'idm_15_price'],
                optimized.at[idx, 'idm_60_price']
            )

        # Initialize tracking for charging needs per time slot
        time_slot_charging_needs = {}
        for time in schedule_times:
            time_slot_charging_needs[time] = {
                'vehicles': [],
                'total_needed_kw': 0
            }

        # STEP 1: First, calculate and record each vehicle's charging needs for each time slot
        print("Calculating charging needs for each vehicle...")
        for vehicle_idx, vehicle in enumerate(eligible_vehicles):
            # Reset vehicle's SOC to initial state
            vehicle['current_soc'] = vehicle['return_soc']
            vehicle['charging_schedule'] = {}

            # Calculate total energy needed
            energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']
            if energy_needed_kwh <= 0:
                continue

            # Find valid charging slots
            charging_slots = optimized[
                (optimized['Time'] >= vehicle['arrival_time']) &
                (optimized['Time'] < vehicle['departure_time'])
                ]
            if len(charging_slots) == 0:
                continue

            # Sort slots by price for optimal charging
            charging_slots = charging_slots.sort_values('dam_price')

            # For each slot, calculate how much power this vehicle could use
            for _, slot in charging_slots.iterrows():
                time = slot['Time']

                # Skip if vehicle is already fully charged
                remaining_energy_kwh = (vehicle['target_soc'] - vehicle['current_soc']) * vehicle['capacity_kwh']
                if remaining_energy_kwh <= 0:
                    break

                # Calculate maximum power for this 15-min period
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 min = 0.25 hour
                charge_amount = min(max_energy_per_period, remaining_energy_kwh)
                charge_kw = charge_amount / 0.25  # Convert back to kW

                if charge_kw > 0:
                    # Record this vehicle's charging need for this time slot
                    time_slot_charging_needs[time]['vehicles'].append({
                        'vehicle': vehicle,
                        'charge_kw': charge_kw,
                        'remaining_energy_kwh': remaining_energy_kwh
                    })
                    time_slot_charging_needs[time]['total_needed_kw'] += charge_kw

        # STEP 2: Process each time slot to allocate charging between DAM and IDM
        print("Allocating charging between DAM and IDM for each time slot...")
        for time, charging_data in time_slot_charging_needs.items():
            if not charging_data['vehicles']:
                continue  # Skip time slots with no vehicles

            total_needed_kw = charging_data['total_needed_kw']
            vehicles = charging_data['vehicles']

            print(f"Time {time}: {len(vehicles)} vehicles need total of {total_needed_kw:.2f} kW")

            # Determine which IDM market to use (15-min or 60-min) based on price
            row = optimized.loc[optimized['Time'] == time].iloc[0]
            use_15_min_idm = row['idm_15_price'] <= row['idm_60_price']

            # CRITICAL PART: Calculate IDM power in 100kW steps for TOTAL fleet power
            # This is the key change - applying the 100kW step to the total, not individual vehicles
            total_idm_kw = int(total_needed_kw / 100) * 100

            # Remainder goes to DAM
            total_dam_kw = total_needed_kw - total_idm_kw

            print(f"  Allocating: {total_idm_kw} kW to IDM and {total_dam_kw:.2f} kW to DAM")

            if total_idm_kw > 0:
                # Update the appropriate IDM counters
                if use_15_min_idm:
                    optimized.loc[optimized['Time'] == time, 'idm_15_charge_kw'] = total_idm_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_15'] = len(vehicles)
                else:
                    optimized.loc[optimized['Time'] == time, 'idm_60_charge_kw'] = total_idm_kw
                    optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm_60'] = len(vehicles)

                optimized.loc[optimized['Time'] == time, 'idm_charge_kw'] = total_idm_kw
                optimized.loc[optimized['Time'] == time, 'total_ev_charging_idm'] = len(vehicles)

            if total_dam_kw > 0:
                optimized.loc[optimized['Time'] == time, 'dam_charge_kw'] = total_dam_kw
                optimized.loc[optimized['Time'] == time, 'total_ev_charging_dam'] = len(vehicles)

            # Total charging power and vehicle count
            optimized.loc[optimized['Time'] == time, 'total_charge_kw'] = total_idm_kw + total_dam_kw
            optimized.loc[optimized['Time'] == time, 'total_ev_charging'] = len(vehicles)

            # STEP 3: Allocate the charging to individual vehicles and update their state
            # Proportionally distribute the IDM and DAM charging to each vehicle
            if total_needed_kw > 0:  # Prevent division by zero
                idm_ratio = total_idm_kw / total_needed_kw
                dam_ratio = total_dam_kw / total_needed_kw

                for vehicle_data in vehicles:
                    vehicle = vehicle_data['vehicle']
                    vehicle_charge_kw = vehicle_data['charge_kw']

                    # Calculate IDM and DAM portions for this vehicle
                    vehicle_idm_kw = vehicle_charge_kw * idm_ratio
                    vehicle_dam_kw = vehicle_charge_kw * dam_ratio

                    # Energy provided in this 15-min slot
                    idm_energy = vehicle_idm_kw * 0.25  # kWh
                    dam_energy = vehicle_dam_kw * 0.25  # kWh
                    total_energy = idm_energy + dam_energy

                    # Update vehicle's SOC
                    vehicle['current_soc'] += total_energy / vehicle['capacity_kwh']

                    # Update vehicle's charging schedule
                    if vehicle_idm_kw > 0 and vehicle_dam_kw > 0:
                        # Mixed charging
                        vehicle['charging_schedule'][time] = {
                            'source': 'MIXED',
                            'kw_dam': vehicle_dam_kw,
                            'kw_idm': vehicle_idm_kw,
                            'idm_type': '15min' if use_15_min_idm else '60min'
                        }
                    elif vehicle_idm_kw > 0:
                        # Only IDM charging
                        vehicle['charging_schedule'][time] = {
                            'source': 'IDM',
                            'kw': vehicle_idm_kw,
                            'idm_type': '15min' if use_15_min_idm else '60min'
                        }
                    elif vehicle_dam_kw > 0:
                        # Only DAM charging
                        vehicle['charging_schedule'][time] = {
                            'source': 'DAM',
                            'kw': vehicle_dam_kw
                        }

        # Calculate costs
        optimized['dam_cost'] = optimized['dam_charge_kw'] * optimized['dam_price'] / 1000 * 0.25
        optimized['idm_15_cost'] = optimized['idm_15_charge_kw'] * optimized['idm_15_price'] / 1000 * 0.25
        optimized['idm_60_cost'] = optimized['idm_60_charge_kw'] * optimized['idm_60_price'] / 1000 * 0.25
        optimized['idm_cost'] = optimized['idm_15_cost'] + optimized['idm_60_cost']

        # Calculate total energy and costs
        total_dam_energy = optimized['dam_charge_kw'].sum() * 0.25
        total_idm_15_energy = optimized['idm_15_charge_kw'].sum() * 0.25
        total_idm_60_energy = optimized['idm_60_charge_kw'].sum() * 0.25
        total_energy = total_dam_energy + total_idm_15_energy + total_idm_60_energy

        total_dam_cost = optimized['dam_cost'].sum()
        total_idm_15_cost = optimized['idm_15_cost'].sum()
        total_idm_60_cost = optimized['idm_60_cost'].sum()
        total_cost = total_dam_cost + total_idm_15_cost + total_idm_60_cost

        # Count how many vehicles used each market type
        vehicles_using_idm15 = set()
        vehicles_using_idm60 = set()
        vehicles_using_dam = set()

        for vehicle in eligible_vehicles:
            for time, schedule in vehicle.get('charging_schedule', {}).items():
                if 'source' in schedule:
                    if schedule['source'] == 'IDM' and 'idm_type' in schedule:
                        if schedule['idm_type'] == '15min':
                            vehicles_using_idm15.add(vehicle['id'])
                        else:
                            vehicles_using_idm60.add(vehicle['id'])
                    elif schedule['source'] == 'MIXED' and 'idm_type' in schedule:
                        if schedule['idm_type'] == '15min':
                            vehicles_using_idm15.add(vehicle['id'])
                        else:
                            vehicles_using_idm60.add(vehicle['id'])
                        vehicles_using_dam.add(vehicle['id'])
                    elif schedule['source'] == 'DAM':
                        vehicles_using_dam.add(vehicle['id'])

        # Print summary
        print("\nOptimized Charging Summary:")
        print(f"Total energy charged: {total_energy:.2f} kWh")
        if total_energy > 0:
            print(f"- DAM: {total_dam_energy:.2f} kWh ({total_dam_energy / total_energy * 100:.1f}%)")
            print(f"- IDM 15-min: {total_idm_15_energy:.2f} kWh ({total_idm_15_energy / total_energy * 100:.1f}%)")
            print(f"- IDM 60-min: {total_idm_60_energy:.2f} kWh ({total_idm_60_energy / total_energy * 100:.1f}%)")

        print(f"\nTotal cost: {total_cost:.2f} €")
        if total_cost > 0:
            print(f"- DAM: {total_dam_cost:.2f} € ({total_dam_cost / total_cost * 100:.1f}%)")
            print(f"- IDM 15-min: {total_idm_15_cost:.2f} € ({total_idm_15_cost / total_cost * 100:.1f}%)")
            print(f"- IDM 60-min: {total_idm_60_cost:.2f} € ({total_idm_60_cost / total_cost * 100:.1f}%)")

        print(f"\nVehicles using markets:")
        print(f"- DAM: {len(vehicles_using_dam)} vehicles")
        print(f"- IDM 15-min: {len(vehicles_using_idm15)} vehicles")
        print(f"- IDM 60-min: {len(vehicles_using_idm60)} vehicles")

        # Store results
        self.optimized_results = optimized
        return optimized

    def optimized_strategy(self, simulation_start, simulation_end=None, options=None, DAM_ALLOCATION=0.05):
        """
        Implementácia optimalizovanej stratégie nabíjania (DAM + IDM)

        Parametre:
        -----------
        simulation_start : datetime
            Počiatočný čas simulácie
        simulation_end : datetime, voliteľné
            Koncový čas simulácie. Ak nie je zadaný, predvolene 24 hodín po začiatku.
        options : dict, voliteľné
            Slovník optimalizačných možností
        DAM_ALLOCATION : float, voliteľné
            Percento energie alokovanej na DAM (predvolene 0.05 alebo 5%)
        """
        print(f"Starting optimized strategy with fleet-level IDM steps (DAM allocation: {DAM_ALLOCATION * 100:.1f}%)")

        # Basic setup
        if simulation_end is None:
            simulation_end = simulation_start + timedelta(hours=24)

        # Create time schedule with 15-minute intervals
        # Use periods instead of end to ensure consistent length
        simulation_periods = int((simulation_end - simulation_start).total_seconds() / 900)  # 900 seconds = 15 minutes
        schedule_times = pd.date_range(
            start=simulation_start,
            periods=simulation_periods,
            freq='15T'
        )

        # Initialize results dataframe
        optimized = pd.DataFrame({
            'Time': schedule_times,
            'total_charge_kw': 0.0,
            'dam_charge_kw': 0.0,
            'idm_charge_kw': 0.0,
            'idm_15_charge_kw': 0.0,
            'idm_60_charge_kw': 0.0,
            'dam_price': 0.0,
            'idm_price': 0.0,
            'idm_15_price': 0.0,
            'idm_60_price': 0.0,
            'dam_cost': 0.0,
            'idm_cost': 0.0,
            'idm_15_cost': 0.0,
            'idm_60_cost': 0.0,
            'total_ev_charging': 0,
            'total_ev_charging_dam': 0,
            'total_ev_charging_idm': 0,
            'total_ev_charging_idm_15': 0,
            'total_ev_charging_idm_60': 0
        })

        print(f"Created schedule with {len(schedule_times)} time slots")
        print(f"Optimized dataframe has {len(optimized)} rows")

        # Get eligible vehicles
        eligible_vehicles = self.get_eligible_vehicles(simulation_start, simulation_end)
        print(f"Found {len(eligible_vehicles)} eligible vehicles")

        # Load price data
        for i in range(len(optimized)):
            time = optimized.iloc[i]['Time']
            hour = time.replace(minute=0, second=0, microsecond=0)

            # Find DAM price (closest hourly match)
            dam_prices_in_range = self.dam_prices[
                (self.dam_prices['Time'] >= hour) &
                (self.dam_prices['Time'] < hour + timedelta(hours=1))
                ]

            if not dam_prices_in_range.empty:
                optimized.iloc[i, optimized.columns.get_loc('dam_price')] = dam_prices_in_range.iloc[0]['cena']
            else:
                # Find closest price if no exact match
                closest_idx = (self.dam_prices['Time'] - hour).abs().idxmin()
                if not pd.isna(closest_idx) and closest_idx < len(self.dam_prices):
                    optimized.iloc[i, optimized.columns.get_loc('dam_price')] = self.dam_prices.iloc[closest_idx][
                        'cena']
                else:
                    optimized.iloc[i, optimized.columns.get_loc('dam_price')] = 50.0  # Default

            # Find IDM 15-min price
            idm_15_prices_in_range = self.idm_15_prices[
                (self.idm_15_prices['Time'] >= time) &
                (self.idm_15_prices['Time'] < time + timedelta(minutes=15))
                ]

            if not idm_15_prices_in_range.empty:
                optimized.iloc[i, optimized.columns.get_loc('idm_15_price')] = idm_15_prices_in_range.iloc[0]['cena']
            else:
                # Use DAM price as fallback
                optimized.iloc[i, optimized.columns.get_loc('idm_15_price')] = optimized.iloc[i]['dam_price']

            # Find IDM 60-min price
            idm_60_prices_in_range = self.idm_60_prices[
                (self.idm_60_prices['Time'] >= hour) &
                (self.idm_60_prices['Time'] < hour + timedelta(hours=1))
                ]

            if not idm_60_prices_in_range.empty:
                optimized.iloc[i, optimized.columns.get_loc('idm_60_price')] = idm_60_prices_in_range.iloc[0]['cena']
            else:
                # Use DAM price as fallback
                optimized.iloc[i, optimized.columns.get_loc('idm_60_price')] = optimized.iloc[i]['dam_price']

            # Set best IDM price (minimum of 15-min and 60-min)
            optimized.iloc[i, optimized.columns.get_loc('idm_price')] = min(
                optimized.iloc[i]['idm_15_price'],
                optimized.iloc[i]['idm_60_price']
            )

        # Create a dictionary to track charging needs for each time slot
        time_slot_needs = {}
        for time in schedule_times:
            time_slot_needs[time] = {
                'vehicles': [],
                'total_energy_kwh': 0.0,
                'total_power_kw': 0.0
            }

        # PHASE 1: Calculate total charging needs for each time slot
        print("\nPhase 1: Calculating charging needs for each time slot...")

        # Reset all vehicles to initial state
        for vehicle in eligible_vehicles:
            vehicle['current_soc'] = vehicle['return_soc']
            vehicle['charging_schedule'] = {}

        # Calculate how much energy each vehicle needs in each time slot
        for vehicle in eligible_vehicles:
            # Total energy needed for this vehicle
            total_energy_needed_kwh = (vehicle['target_soc'] - vehicle['return_soc']) * vehicle['capacity_kwh']

            if total_energy_needed_kwh <= 0:
                continue  # Skip if no charging needed

            # Get time slots when this vehicle is available for charging
            valid_times = [t for t in schedule_times if t >= vehicle['arrival_time'] and t < vehicle['departure_time']]

            if not valid_times:
                continue  # Skip if no valid charging times

            # Sort charging slots by price (use DAM price for sorting)
            time_prices = [(t, optimized.loc[optimized['Time'] == t, 'dam_price'].iloc[0]) for t in valid_times]
            time_prices.sort(key=lambda x: x[1])  # Sort by price

            # Calculate remaining energy and distribute across time slots
            remaining_energy_kwh = total_energy_needed_kwh
            for time, _ in time_prices:
                if remaining_energy_kwh <= 0:
                    break

                # Maximum energy this vehicle can charge in this 15-min period
                max_energy_per_period = vehicle['max_charge_kw'] * 0.25  # 15 min = 0.25 hour
                charge_amount_kwh = min(max_energy_per_period, remaining_energy_kwh)
                charge_power_kw = charge_amount_kwh / 0.25  # Convert back to kW

                if charge_amount_kwh > 0:
                    # Record this vehicle's charging need
                    time_slot_needs[time]['vehicles'].append({
                        'vehicle': vehicle,
                        'energy_kwh': charge_amount_kwh,
                        'power_kw': charge_power_kw
                    })

                    # Update totals for this time slot
                    time_slot_needs[time]['total_energy_kwh'] += charge_amount_kwh
                    time_slot_needs[time]['total_power_kw'] += charge_power_kw

                    # Update remaining energy
                    remaining_energy_kwh -= charge_amount_kwh

        # PHASE 2: Allocate between DAM and IDM for each time slot
        print("\nPhase 2: Allocating between DAM and IDM for each time slot...")

        for i, time in enumerate(schedule_times):
            slot_needs = time_slot_needs[time]

            if not slot_needs['vehicles']:
                continue  # Skip if no vehicles in this time slot

            total_power_kw = slot_needs['total_power_kw']

            # First apply DAM_ALLOCATION to determine initial split
            dam_target_kw = total_power_kw * DAM_ALLOCATION
            idm_target_kw = total_power_kw - dam_target_kw

            # Round IDM power to nearest 100kW step (IDM requirement)
            idm_actual_kw = int(idm_target_kw / 100) * 100

            # Remainder goes to DAM
            dam_actual_kw = total_power_kw - idm_actual_kw

            print(
                f"Time {time}: Total: {total_power_kw:.2f} kW -> DAM: {dam_actual_kw:.2f} kW, IDM: {idm_actual_kw} kW")

            # Determine which IDM market to use (15-min or 60-min)
            idm_15_price = optimized.iloc[i]['idm_15_price']
            idm_60_price = optimized.iloc[i]['idm_60_price']
            use_15_min_idm = idm_15_price <= idm_60_price

            # Update optimized DataFrame - Use iloc to avoid index issues
            if idm_actual_kw > 0:
                if use_15_min_idm:
                    optimized.iloc[i, optimized.columns.get_loc('idm_15_charge_kw')] = idm_actual_kw
                    optimized.iloc[i, optimized.columns.get_loc('total_ev_charging_idm_15')] = len(
                        slot_needs['vehicles'])
                else:
                    optimized.iloc[i, optimized.columns.get_loc('idm_60_charge_kw')] = idm_actual_kw
                    optimized.iloc[i, optimized.columns.get_loc('total_ev_charging_idm_60')] = len(
                        slot_needs['vehicles'])

                optimized.iloc[i, optimized.columns.get_loc('idm_charge_kw')] = idm_actual_kw
                optimized.iloc[i, optimized.columns.get_loc('total_ev_charging_idm')] = len(slot_needs['vehicles'])

            if dam_actual_kw > 0:
                optimized.iloc[i, optimized.columns.get_loc('dam_charge_kw')] = dam_actual_kw
                optimized.iloc[i, optimized.columns.get_loc('total_ev_charging_dam')] = len(slot_needs['vehicles'])

            optimized.iloc[i, optimized.columns.get_loc('total_charge_kw')] = total_power_kw
            optimized.iloc[i, optimized.columns.get_loc('total_ev_charging')] = len(slot_needs['vehicles'])

            # PHASE 3: Distribute back to individual vehicles
            if total_power_kw > 0:  # Avoid division by zero
                idm_ratio = idm_actual_kw / total_power_kw
                dam_ratio = dam_actual_kw / total_power_kw

                for vehicle_data in slot_needs['vehicles']:
                    vehicle = vehicle_data['vehicle']
                    vehicle_power_kw = vehicle_data['power_kw']

                    # Calculate how much power goes to each market for this vehicle
                    vehicle_idm_kw = vehicle_power_kw * idm_ratio
                    vehicle_dam_kw = vehicle_power_kw * dam_ratio

                    # Calculate energy charged
                    idm_energy_kwh = vehicle_idm_kw * 0.25  # 15 min = 0.25 hour
                    dam_energy_kwh = vehicle_dam_kw * 0.25
                    total_energy_kwh = idm_energy_kwh + dam_energy_kwh

                    # Update vehicle SOC
                    vehicle['current_soc'] += total_energy_kwh / vehicle['capacity_kwh']

                    # Update vehicle charging schedule
                    if vehicle_idm_kw > 0 and vehicle_dam_kw > 0:
                        # Mixed charging
                        vehicle['charging_schedule'][time] = {
                            'source': 'MIXED',
                            'kw_dam': vehicle_dam_kw,
                            'kw_idm': vehicle_idm_kw,
                            'idm_type': '15min' if use_15_min_idm else '60min'
                        }
                    elif vehicle_idm_kw > 0:
                        # Only IDM
                        vehicle['charging_schedule'][time] = {
                            'source': 'IDM',
                            'kw': vehicle_idm_kw,
                            'idm_type': '15min' if use_15_min_idm else '60min'
                        }
                    elif vehicle_dam_kw > 0:
                        # Only DAM
                        vehicle['charging_schedule'][time] = {
                            'source': 'DAM',
                            'kw': vehicle_dam_kw
                        }

        # Calculate costs
        for i in range(len(optimized)):
            dam_charge = optimized.iloc[i]['dam_charge_kw']
            idm_15_charge = optimized.iloc[i]['idm_15_charge_kw']
            idm_60_charge = optimized.iloc[i]['idm_60_charge_kw']

            dam_price = optimized.iloc[i]['dam_price']
            idm_15_price = optimized.iloc[i]['idm_15_price']
            idm_60_price = optimized.iloc[i]['idm_60_price']

            # Calculate costs (kW * €/MWh * 0.25h / 1000 = €)
            optimized.iloc[i, optimized.columns.get_loc('dam_cost')] = dam_charge * dam_price / 1000 * 0.25
            optimized.iloc[i, optimized.columns.get_loc('idm_15_cost')] = idm_15_charge * idm_15_price / 1000 * 0.25
            optimized.iloc[i, optimized.columns.get_loc('idm_60_cost')] = idm_60_charge * idm_60_price / 1000 * 0.25
            optimized.iloc[i, optimized.columns.get_loc('idm_cost')] = (
                    optimized.iloc[i]['idm_15_cost'] + optimized.iloc[i]['idm_60_cost']
            )

        # Calculate total energy and costs
        total_dam_energy = optimized['dam_charge_kw'].sum() * 0.25  # kWh
        total_idm_15_energy = optimized['idm_15_charge_kw'].sum() * 0.25  # kWh
        total_idm_60_energy = optimized['idm_60_charge_kw'].sum() * 0.25  # kWh
        total_energy = total_dam_energy + total_idm_15_energy + total_idm_60_energy

        total_dam_cost = optimized['dam_cost'].sum()
        total_idm_15_cost = optimized['idm_15_cost'].sum()
        total_idm_60_cost = optimized['idm_60_cost'].sum()
        total_cost = total_dam_cost + total_idm_15_cost + total_idm_60_cost

        # Print summary
        print("\nOptimized Charging Summary:")
        print(f"Total energy charged: {total_energy:.2f} kWh")
        if total_energy > 0:
            print(f"- DAM: {total_dam_energy:.2f} kWh ({total_dam_energy / total_energy * 100:.1f}%)")
            print(f"- IDM 15-min: {total_idm_15_energy:.2f} kWh ({total_idm_15_energy / total_energy * 100:.1f}%)")
            print(f"- IDM 60-min: {total_idm_60_energy:.2f} kWh ({total_idm_60_energy / total_energy * 100:.1f}%)")

        print(f"\nTotal cost: {total_cost:.2f} €")
        if total_cost > 0:
            print(f"- DAM: {total_dam_cost:.2f} € ({total_dam_cost / total_cost * 100:.1f}%)")
            print(f"- IDM 15-min: {total_idm_15_cost:.2f} € ({total_idm_15_cost / total_cost * 100:.1f}%)")
            print(f"- IDM 60-min: {total_idm_60_cost:.2f} € ({total_idm_60_cost / total_cost * 100:.1f}%)")

        # Store results
        self.optimized_results = optimized
        return optimized

    def aggregate_hourly_results(self, results):
        """
        Agregácia 15-minútových výsledkov na hodinové pre porovnanie

        Parametre:
        ----------
        results : DataFrame
            DataFrame s výsledkami v 15-minútovej granularite

        Výstup:
        -------
        DataFrame s agregovanými hodinovými výsledkami alebo pôvodné výsledky, ak už sú hodinové
        """
        if results is None:
            return None

        # Kontrola, či ide o 15-minútové údaje, ktoré treba agregovať
        # Namiesto spoliehania sa na frekvenčný atribút kontrolujeme skutočné časové rozdiely
        if len(results) > 1:
            time_diff = results['Time'].iloc[1] - results['Time'].iloc[0]
            if time_diff.total_seconds() == 900:  # 15 minút = 900 sekúnd
                # Vytvorenie kópie a nastavenie datetime ako indexu
                hourly_results = results.copy()
                hourly_results['hour'] = hourly_results['Time'].dt.floor('H')

                # Zoskupenie podľa hodiny a agregácia
                agg_dict = {
                    'total_charge_kw': 'mean',
                    'total_ev_charging': 'mean'
                }

                # Pridanie stĺpcov, ktoré môžu existovať len v optimalizovaných výsledkoch
                if 'dam_charge_kw' in hourly_results.columns:
                    agg_dict['dam_charge_kw'] = 'mean'
                if 'idm_charge_kw' in hourly_results.columns:
                    agg_dict['idm_charge_kw'] = 'mean'
                if 'idm_15_charge_kw' in hourly_results.columns:
                    agg_dict['idm_15_charge_kw'] = 'mean'
                if 'idm_60_charge_kw' in hourly_results.columns:
                    agg_dict['idm_60_charge_kw'] = 'mean'
                if 'dam_price' in hourly_results.columns:
                    agg_dict['dam_price'] = 'mean'
                if 'idm_price' in hourly_results.columns:
                    agg_dict['idm_price'] = 'mean'
                if 'idm_15_price' in hourly_results.columns:
                    agg_dict['idm_15_price'] = 'mean'
                if 'idm_60_price' in hourly_results.columns:
                    agg_dict['idm_60_price'] = 'mean'
                if 'dam_cost' in hourly_results.columns:
                    agg_dict['dam_cost'] = 'sum'
                if 'idm_cost' in hourly_results.columns:
                    agg_dict['idm_cost'] = 'sum'
                if 'idm_15_cost' in hourly_results.columns:
                    agg_dict['idm_15_cost'] = 'sum'
                if 'idm_60_cost' in hourly_results.columns:
                    agg_dict['idm_60_cost'] = 'sum'
                if 'total_ev_charging_dam' in hourly_results.columns:
                    agg_dict['total_ev_charging_dam'] = 'mean'
                if 'total_ev_charging_idm' in hourly_results.columns:
                    agg_dict['total_ev_charging_idm'] = 'mean'
                if 'total_ev_charging_idm_15' in hourly_results.columns:
                    agg_dict['total_ev_charging_idm_15'] = 'mean'
                if 'total_ev_charging_idm_60' in hourly_results.columns:
                    agg_dict['total_ev_charging_idm_60'] = 'mean'

                aggregated = hourly_results.groupby('hour').agg(agg_dict).reset_index()

                # Premenovanie stĺpca hour na Time pre konzistentnosť
                aggregated.rename(columns={'hour': 'Time'}, inplace=True)

                return aggregated

        # Ak nejde o 15-minútové údaje alebo ich nemožno agregovať, vrátime pôvodné
        return results

    def visualize_results(self):
        """Vygenerovanie všetkých požadovaných vizualizácií"""
        if self.baseline_results is None or self.optimized_results is None:
            print("Žiadne výsledky na vizualizáciu. Najprv spustite stratégie.")
            print(f"Baseline výsledky: {'K dispozícii' if self.baseline_results is not None else 'Žiadne'}")
            print(f"Optimalizované výsledky: {'K dispozícii' if self.optimized_results is not None else 'Žiadne'}")
            return

        # Agregácia optimalizovaných výsledkov na hodinové pre porovnanie
        hourly_optimized = self.aggregate_hourly_results(self.optimized_results)

        # 1. Časové série nákupov pre obe stratégie
        self._plot_time_series_purchases()

        # 2. Hodinové porovnanie nabíjacieho výkonu
        self._plot_hourly_charging_power(self.baseline_results, hourly_optimized)

        # 3. Hodinové porovnanie nákladov na elektrinu
        self._plot_hourly_electricity_price(self.baseline_results, hourly_optimized)

    def _plot_time_series_purchases(self):
        """Graf časových sérií nákupov v čase pre obe stratégie"""
        plt.figure(figsize=(14, 12))  # Väčšia výška pre dodatočné subplot-y

        # Vytvorenie farebnej schémy pre pozadia víkendov
        weekend_color = 'lightgray'
        weekend_alpha = 0.2

        # Definícia slovenských názvov dní v týždni
        sk_day_names = {
            0: 'Po', 1: 'Ut', 2: 'St', 3: 'Št', 4: 'Pi', 5: 'So', 6: 'Ne'
        }

        # Graf baseline stratégie (len DAM)
        plt.subplot(3, 1, 1)
        plt.title('Základná stratégia (len DAM)', fontsize=14)

        # Vykreslenie pozadia pre víkendy v základnej stratégii
        prev_weekend = False
        weekend_start = None

        for i, time in enumerate(self.baseline_results['Time']):
            is_weekend = time.weekday() >= 5  # Sobota (5) alebo Nedeľa (6)

            # Začiatok víkendu
            if is_weekend and not prev_weekend:
                weekend_start = time
                prev_weekend = True

            # Koniec víkendu
            elif not is_weekend and prev_weekend:
                if weekend_start:
                    plt.axvspan(weekend_start, time, alpha=weekend_alpha, color=weekend_color,
                                label='Víkend' if i == 1 else '')
                prev_weekend = False

        # Zachytenie prípadu, keď víkend končí na konci dát
        if prev_weekend and weekend_start:
            plt.axvspan(weekend_start, self.baseline_results['Time'].iloc[-1] + timedelta(hours=1),
                        alpha=weekend_alpha, color=weekend_color)

        # Vykreslenie hlavných dát
        plt.plot(self.baseline_results['Time'], self.baseline_results['total_charge_kw'], '-o',
                 color='blue', markersize=4, label='Nabíjací výkon (kW)')
        plt.plot(self.baseline_results['Time'], self.baseline_results['dam_price'], '-x',
                 color='red', markersize=4, label='Cena (€/MWh)')

        # Vytvorenie lepších popiskov x-osi s dňom v týždni
        time_labels = []
        for t in self.baseline_results['Time']:
            day_name = sk_day_names[t.weekday()]
            if t.hour == 0:  # Začiatok dňa
                time_labels.append(f"{day_name} {t.day}.{t.month}")
            elif t.hour % 6 == 0:  # Každých 6 hodín
                time_labels.append(f"{t.hour}:00")
            else:
                time_labels.append('')

        # Nastavenie popiskov x-osi
        if len(self.baseline_results['Time']) > 30:
            # Ak je príliš veľa popiskov, zobrazíme len niektoré
            plt.xticks(self.baseline_results['Time'][::4], time_labels[::4], rotation=45, ha='right')
        else:
            plt.xticks(self.baseline_results['Time'], time_labels, rotation=45, ha='right')

        plt.xlabel('Čas')
        plt.ylabel('Výkon [kW] / Cena [€/MWh]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')

        # Graf optimalizovanej stratégie (DAM + celkové IDM)
        plt.subplot(3, 1, 2)
        plt.title('Optimalizovaná stratégia (DAM + celkové IDM)', fontsize=14)

        # Vykreslenie pozadia pre víkendy v optimalizovanej stratégii
        prev_weekend = False
        weekend_start = None

        for i, time in enumerate(self.optimized_results['Time']):
            is_weekend = time.weekday() >= 5

            # Začiatok víkendu
            if is_weekend and not prev_weekend:
                weekend_start = time
                prev_weekend = True

            # Koniec víkendu
            elif not is_weekend and prev_weekend:
                if weekend_start:
                    plt.axvspan(weekend_start, time, alpha=weekend_alpha, color=weekend_color,
                                label='Víkend' if i == 1 else '')
                prev_weekend = False

        # Zachytenie prípadu, keď víkend končí na konci dát
        if prev_weekend and weekend_start:
            plt.axvspan(weekend_start, self.optimized_results['Time'].iloc[-1] + timedelta(minutes=15),
                        alpha=weekend_alpha, color=weekend_color)

        # Vykreslenie hlavných dát optimalizovanej stratégie
        plt.plot(self.optimized_results['Time'], self.optimized_results['dam_charge_kw'], '-o',
                 color='blue', markersize=2, alpha=0.7, label='DAM nabíjanie (kW)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_charge_kw'], '-o',
                 color='green', markersize=2, alpha=0.7, label='IDM celkové nabíjanie (kW)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['dam_price'], '-x',
                 color='red', markersize=2, alpha=0.5, label='DAM cena (€/MWh)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_price'], '-x',
                 color='purple', markersize=2, alpha=0.5, label='IDM cena (€/MWh)')

        # Vytvorenie popiskov x-osi
        time_labels = []
        for t in self.optimized_results['Time']:
            day_name = sk_day_names[t.weekday()]
            if t.hour == 0 and t.minute == 0:  # Začiatok dňa
                time_labels.append(f"{day_name} {t.day}.{t.month}")
            elif t.hour % 6 == 0 and t.minute == 0:  # Každých 6 hodín
                time_labels.append(f"{t.hour}:00")
            else:
                time_labels.append('')

        # Nastavenie popiskov x-osi
        if len(self.optimized_results['Time']) > 100:
            plt.xticks(self.optimized_results['Time'][::16], time_labels[::16], rotation=45, ha='right')
        else:
            plt.xticks(self.optimized_results['Time'][::4], time_labels[::4], rotation=45, ha='right')

        plt.xlabel('Čas')
        plt.ylabel('Výkon [kW] / Cena [€/MWh]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')

        # Graf optimalizovanej stratégie (IDM podľa typu)
        plt.subplot(3, 1, 3)
        plt.title('Optimalizovaná stratégia (IDM podľa typu)', fontsize=14)

        # Vykreslenie pozadia pre víkendy v rozdelenom grafe IDM
        prev_weekend = False
        weekend_start = None

        for i, time in enumerate(self.optimized_results['Time']):
            is_weekend = time.weekday() >= 5

            # Začiatok víkendu
            if is_weekend and not prev_weekend:
                weekend_start = time
                prev_weekend = True

            # Koniec víkendu
            elif not is_weekend and prev_weekend:
                if weekend_start:
                    plt.axvspan(weekend_start, time, alpha=weekend_alpha, color=weekend_color,
                                label='Víkend' if i == 1 else '')
                prev_weekend = False

        # Zachytenie prípadu, keď víkend končí na konci dát
        if prev_weekend and weekend_start:
            plt.axvspan(weekend_start, self.optimized_results['Time'].iloc[-1] + timedelta(minutes=15),
                        alpha=weekend_alpha, color=weekend_color)

        # Vykreslenie rozdelených IDM dát
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_15_charge_kw'], '-o',
                 color='green', markersize=2, alpha=0.7, label='IDM 15-min nabíjanie (kW)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_60_charge_kw'], '-o',
                 color='orange', markersize=2, alpha=0.7, label='IDM 60-min nabíjanie (kW)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_15_price'], '-x',
                 color='green', markersize=2, alpha=0.3, label='IDM 15-min cena (€/MWh)')
        plt.plot(self.optimized_results['Time'], self.optimized_results['idm_60_price'], '-x',
                 color='orange', markersize=2, alpha=0.3, label='IDM 60-min cena (€/MWh)')

        # Nastavenie popiskov x-osi
        if len(self.optimized_results['Time']) > 100:
            plt.xticks(self.optimized_results['Time'][::16], time_labels[::16], rotation=45, ha='right')
        else:
            plt.xticks(self.optimized_results['Time'][::4], time_labels[::4], rotation=45, ha='right')

        plt.xlabel('Čas')
        plt.ylabel('Výkon [kW] / Cena [€/MWh]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')

        # Pridanie poznámky o režime práce
        mode_text = 'Režim: Len pracovné dni' if hasattr(self,
                                                         'workdays_only') and self.workdays_only else 'Režim: Všetky dni'
        plt.figtext(0.5, 0.01, mode_text, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Upravené, aby sa zmestil text dole
        plt.savefig('plots/time_series_purchases.png', dpi=300)
        plt.show()

    def _plot_hourly_charging_power(self, baseline, optimized):
        """Graf hodinového porovnania nabíjacieho výkonu medzi dvoma stratégiami"""
        plt.figure(figsize=(14, 6))

        # Extrakcia času pre štítky x-osi
        times = [time.strftime('%m-%d %H:%M') for time in baseline['Time']]
        x = np.arange(len(times))

        # Nastavenie šírky stĺpcov
        width = 0.35

        # Vytvorenie zoskupených stĺpcov
        plt.bar(x - width / 2, baseline['total_charge_kw'], width, label='DAM (Základná stratégia)',
                color='blue', alpha=0.7)

        # Rozdelenie optimalizovanej stratégie na DAM a IDM komponenty (s ďalším rozdelením IDM)
        plt.bar(x + width / 2, optimized['dam_charge_kw'], width, label='DAM (Optimalizované)',
                color='blue', alpha=0.3)
        plt.bar(x + width / 2, optimized['idm_15_charge_kw'], width,
                bottom=optimized['dam_charge_kw'],
                label='IDM 15-min', color='green', alpha=0.7)
        plt.bar(x + width / 2, optimized['idm_60_charge_kw'], width,
                bottom=optimized['dam_charge_kw'] + optimized['idm_15_charge_kw'],
                label='IDM 60-min', color='orange', alpha=0.7)

        plt.xlabel('Čas')
        plt.ylabel('Nabíjací výkon [kW]')
        plt.title('Hodinové porovnanie nabíjacieho výkonu')

        # Zobrazenie iba podmnožiny štítkov, ak je ich príliš veľa
        if len(times) > 24:
            plt.xticks(x[::4], times[::4], rotation=45)
        else:
            plt.xticks(x, times, rotation=45)

        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/hourly_charging_power.png', dpi=300)
        plt.show()

    def _plot_hourly_electricity_price(self, baseline, optimized):
        """Graf hodinového porovnania nákladov na elektrinu medzi dvoma stratégiami"""
        plt.figure(figsize=(14, 6))

        # Extrakcia času pre štítky x-osi
        times = [time.strftime('%m-%d %H:%M') for time in baseline['Time']]
        x = np.arange(len(times))

        # Nastavenie šírky stĺpcov
        width = 0.35

        # Výpočet hodinových nákladov
        baseline_hourly_cost = baseline['dam_cost']
        optimized_dam_cost = optimized['dam_cost']
        optimized_idm_15_cost = optimized['idm_15_cost']
        optimized_idm_60_cost = optimized['idm_60_cost']
        optimized_total_cost = optimized_dam_cost + optimized_idm_15_cost + optimized_idm_60_cost

        # Vytvorenie zoskupených stĺpcov pre náklady
        plt.bar(x - width / 2, baseline_hourly_cost, width, label='DAM (Základná stratégia)',
                color='blue', alpha=0.7)

        # Rozdelenie optimalizovanej stratégie na DAM a IDM komponenty (s ďalším rozdelením IDM)
        plt.bar(x + width / 2, optimized_dam_cost, width, label='DAM (Optimalizované)',
                color='blue', alpha=0.3)
        plt.bar(x + width / 2, optimized_idm_15_cost, width, bottom=optimized_dam_cost,
                label='IDM 15-min', color='green', alpha=0.7)
        plt.bar(x + width / 2, optimized_idm_60_cost, width,
                bottom=optimized_dam_cost + optimized_idm_15_cost,
                label='IDM 60-min', color='orange', alpha=0.7)

        plt.xlabel('Čas')
        plt.ylabel('Náklady [€]')
        plt.title('Hodinové porovnanie nákladov ')

        # Zobrazenie iba podmnožiny štítkov, ak je ich príliš veľa
        if len(times) > 24:
            plt.xticks(x[::4], times[::4], rotation=45)
        else:
            plt.xticks(x, times, rotation=45)

        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)

        # Pridanie celkových nákladov ako text
        baseline_total = baseline_hourly_cost.sum()
        optimized_dam_total = optimized_dam_cost.sum()
        optimized_idm_15_total = optimized_idm_15_cost.sum()
        optimized_idm_60_total = optimized_idm_60_cost.sum()
        optimized_total = optimized_total_cost.sum()
        savings = baseline_total - optimized_total
        savings_pct = (savings / baseline_total) * 100 if baseline_total > 0 else 0

        plt.figtext(0.5, 0.01,
                    f'Celkové náklady - Základná: {baseline_total:.2f} €, Optimalizovaná: {optimized_total:.2f} €\n'
                    f'Rozdelenie optimalizovanej - DAM: {optimized_dam_total:.2f} €, IDM 15-min: {optimized_idm_15_total:.2f} €, IDM 60-min: {optimized_idm_60_total:.2f} €\n'
                    f'Úspory: {savings:.2f} € ({savings_pct:.1f}%)',
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 1])  # Upravený rect pre text
        plt.savefig('plots/hourly_electricity_cost.png', dpi=300)
        plt.show()

    def summarize_results(self):
        """Tlač súhrnu výsledkov"""
        if self.baseline_results is None or self.optimized_results is None:
            print("Žiadne výsledky na zhrnutie. Najprv spustite stratégie.")
            print(f"Baseline výsledky: {'K dispozícii' if self.baseline_results is not None else 'Žiadne'}")
            print(f"Optimalizované výsledky: {'K dispozícii' if self.optimized_results is not None else 'Žiadne'}")
            return

        # Vypočítanie simulačného obdobia v dňoch
        simulation_start = self.baseline_results['Time'].min()
        simulation_end = self.baseline_results['Time'].max()
        simulation_days = (simulation_end - simulation_start).total_seconds() / (24 * 3600) + (
                    1 / 24)  # Pridanie jednej hodiny, aby sa predišlo zaokrúhľovaniu nadol
        simulation_days = max(1, round(simulation_days))  # Zabezpečenie aspoň 1 dňa

        print(f"Simulačné obdobie: {simulation_start} až {simulation_end} ({simulation_days} dní)")

        # Agregácia optimalizovaných výsledkov na hodinové pre porovnanie
        hourly_optimized = self.aggregate_hourly_results(self.optimized_results)

        # Výpočet celkových nákladov
        baseline_total_cost = self.baseline_results['dam_cost'].sum()
        optimized_dam_cost = hourly_optimized['dam_cost'].sum()
        optimized_idm_15_cost = hourly_optimized[
            'idm_15_cost'].sum() if 'idm_15_cost' in hourly_optimized.columns else 0
        optimized_idm_60_cost = hourly_optimized[
            'idm_60_cost'].sum() if 'idm_60_cost' in hourly_optimized.columns else 0
        optimized_idm_cost = optimized_idm_15_cost + optimized_idm_60_cost
        optimized_total_cost = optimized_dam_cost + optimized_idm_cost

        # Výpočet úspor
        savings = baseline_total_cost - optimized_total_cost
        savings_pct = (savings / baseline_total_cost) * 100 if baseline_total_cost > 0 else 0

        # Výpočet spotreby energie
        baseline_energy = self.baseline_results['total_charge_kw'].sum()
        optimized_dam_energy = hourly_optimized[
            'dam_charge_kw'].sum() if 'dam_charge_kw' in hourly_optimized.columns else 0
        optimized_idm_15_energy = hourly_optimized[
            'idm_15_charge_kw'].sum() if 'idm_15_charge_kw' in hourly_optimized.columns else 0
        optimized_idm_60_energy = hourly_optimized[
            'idm_60_charge_kw'].sum() if 'idm_60_charge_kw' in hourly_optimized.columns else 0
        optimized_total_energy = optimized_dam_energy + optimized_idm_15_energy + optimized_idm_60_energy

        # Výpočet priemerných cien
        baseline_avg_price = baseline_total_cost / (baseline_energy / 1000) if baseline_energy > 0 else 0
        optimized_dam_avg_price = optimized_dam_cost / (optimized_dam_energy / 1000) if optimized_dam_energy > 0 else 0
        optimized_idm_15_avg_price = optimized_idm_15_cost / (
                    optimized_idm_15_energy / 1000) if optimized_idm_15_energy > 0 else 0
        optimized_idm_60_avg_price = optimized_idm_60_cost / (
                    optimized_idm_60_energy / 1000) if optimized_idm_60_energy > 0 else 0
        optimized_total_avg_price = optimized_total_cost / (
                    optimized_total_energy / 1000) if optimized_total_energy > 0 else 0

        # Výpočet denných a na vozidlo prepočítaných metrík
        daily_baseline_cost = baseline_total_cost / simulation_days
        daily_optimized_cost = optimized_total_cost / simulation_days
        daily_savings = savings / simulation_days

        vehicle_days = self.fleet_size * simulation_days
        per_vehicle_baseline_cost = baseline_total_cost / vehicle_days
        per_vehicle_optimized_cost = optimized_total_cost / vehicle_days
        per_vehicle_savings = savings / vehicle_days

        # Výpočet priemerného SOC vozidiel pri odchode
        avg_departure_soc = sum(v['current_soc'] for v in self.vehicles) / len(self.vehicles)
        min_departure_soc = min(v['current_soc'] for v in self.vehicles)

        # Tlač súhrnu
        print("\n" + "=" * 70)
        print(
            f"VÝSLEDKY OPTIMALIZÁCIE NABÍJANIA FLOTILY (Veľkosť flotily: {self.fleet_size}, Trvanie: {simulation_days} dní)")
        print("=" * 70)
        print(f"{'Parameter':<30} {'Základná stratégia':<20} {'Optimalizovaná':<20} {'Úspory':<20}")
        print("-" * 70)
        print(
            f"{'Celkové náklady na nabíjanie (€)':<30} {baseline_total_cost:<20.2f} {optimized_total_cost:<20.2f} {savings:<20.2f}")
        print(
            f"{'DAM náklady (€)':<30} {baseline_total_cost:<20.2f} {optimized_dam_cost:<20.2f} {baseline_total_cost - optimized_dam_cost:<20.2f}")
        print(
            f"{'IDM 15-min náklady (€)':<30} {'0.00':<20} {optimized_idm_15_cost:<20.2f} {-optimized_idm_15_cost:<20.2f}")
        print(
            f"{'IDM 60-min náklady (€)':<30} {'0.00':<20} {optimized_idm_60_cost:<20.2f} {-optimized_idm_60_cost:<20.2f}")
        print("-" * 70)
        print(
            f"{'Denné náklady (€/deň)':<30} {daily_baseline_cost:<20.2f} {daily_optimized_cost:<20.2f} {daily_savings:<20.2f}")
        print(
            f"{'Náklady na vozidlo-deň (€)':<30} {per_vehicle_baseline_cost:<20.2f} {per_vehicle_optimized_cost:<20.2f} {per_vehicle_savings:<20.2f}")
        print(f"{'Úspory (%)':<30} {'-':<20} {savings_pct:<20.1f} {'-':<20}")
        print("-" * 70)
        print(
            f"{'Celková nabitá energia (kWh)':<30} {baseline_energy:<20.2f} {optimized_total_energy:<20.2f} {'-':<20}")
        print(f"{'DAM nabitá energia (kWh)':<30} {baseline_energy:<20.2f} {optimized_dam_energy:<20.2f} {'-':<20}")
        print(f"{'IDM 15-min energia (kWh)':<30} {'0.00':<20} {optimized_idm_15_energy:<20.2f} {'-':<20}")
        print(f"{'IDM 60-min energia (kWh)':<30} {'0.00':<20} {optimized_idm_60_energy:<20.2f} {'-':<20}")
        print("-" * 70)
        print(
            f"{'Priemerná cena (€/MWh)':<30} {baseline_avg_price:<20.2f} {optimized_total_avg_price:<20.2f} {baseline_avg_price - optimized_total_avg_price:<20.2f}")
        print(
            f"{'DAM priemerná cena (€/MWh)':<30} {baseline_avg_price:<20.2f} {optimized_dam_avg_price:<20.2f} {'-':<20}")
        print(f"{'IDM 15-min priem. cena (€/MWh)':<30} {'-':<20} {optimized_idm_15_avg_price:<20.2f} {'-':<20}")
        print(f"{'IDM 60-min priem. cena (€/MWh)':<30} {'-':<20} {optimized_idm_60_avg_price:<20.2f} {'-':<20}")
        print("-" * 70)
        print(
            f"{'Denná energia (kWh/deň)':<30} {baseline_energy / simulation_days:<20.2f} {optimized_total_energy / simulation_days:<20.2f} {'-':<20}")
        print(
            f"{'Energia na vozidlo-deň (kWh)':<30} {baseline_energy / vehicle_days:<20.2f} {optimized_total_energy / vehicle_days:<20.2f} {'-':<20}")
        print("-" * 70)
        print(
            f"{'Priemerný SOC pri odchode (%)':<30} {avg_departure_soc * 100:<20.1f} {avg_departure_soc * 100:<20.1f} {'-':<20}")
        print(
            f"{'Minimálny SOC pri odchode (%)':<30} {min_departure_soc * 100:<20.1f} {min_departure_soc * 100:<20.1f} {'-':<20}")
        print(
            f"{'Cieľový SOC (%)':<30} {self.min_soc_target * 100:<20.1f} {self.min_soc_target * 100:<20.1f} {'-':<20}")
        print("=" * 70)

        # Kontrola, či všetky vozidlá dosiahli cieľový SOC
        all_reached_target = all(v['current_soc'] >= v['target_soc'] for v in self.vehicles)
        if all_reached_target:
            print("✅ Všetky vozidlá dosiahli cieľový SOC.")
        else:
            print("⚠️ Nie všetky vozidlá dosiahli cieľový SOC!")
            count = 0
            for v in self.vehicles:
                if v['current_soc'] < v['target_soc']:
                    count += 1
                    if count <= 5:  # Zobrazí iba prvých 5, aby sa nezahltil výstup
                        print(
                            f"  Vozidlo {v['id']}: Aktuálny SOC {v['current_soc'] * 100:.1f}%, Cieľový SOC {v['target_soc'] * 100:.1f}%")
            if count > 5:
                print(f"  ... a ďalších {count - 5} vozidiel")
        print("=" * 70)

        # Tlač štatistík využitia trhov
        idm_type_count = {}
        for vehicle in self.vehicles:
            for time, schedule in vehicle['charging_schedule'].items():
                if isinstance(schedule, dict) and 'source' in schedule:
                    if schedule['source'] == 'IDM' and 'idm_type' in schedule:
                        idm_type = schedule['idm_type']
                        idm_type_count[idm_type] = idm_type_count.get(idm_type, 0) + 1
                    elif schedule['source'] == 'MIXED' and 'idm_type' in schedule:
                        idm_type = schedule['idm_type']
                        idm_type_count[idm_type] = idm_type_count.get(idm_type, 0) + 1

        if idm_type_count:
            print("Štatistiky využitia IDM trhov:")
            for idm_type, count in idm_type_count.items():
                print(f"  {idm_type}: {count} nabíjacích slotov")
        print("=" * 70)

    def export_results(self, format='csv'):
        """
        Export výsledkov do požadovaného formátu

        Parametre:
        -----------
        format : str
            Formát exportu ('csv', 'excel', alebo 'json')
        """
        # Vytvorenie adresára pre výsledky, ak neexistuje
        os.makedirs('results', exist_ok=True)

        if format == 'csv':
            if self.baseline_results is not None:
                self.baseline_results.to_csv('results/baseline_results.csv', index=False)
                print("Baseline výsledky exportované do results/baseline_results.csv")
            if self.optimized_results is not None:
                self.optimized_results.to_csv('results/optimized_results.csv', index=False)
                print("Optimalizované výsledky exportované do results/optimized_results.csv")
        elif format == 'excel':
            with pd.ExcelWriter('results/simulation_results.xlsx') as writer:
                if self.baseline_results is not None:
                    self.baseline_results.to_excel(writer, sheet_name='Baseline', index=False)
                if self.optimized_results is not None:
                    self.optimized_results.to_excel(writer, sheet_name='Optimized', index=False)
                print("Výsledky exportované do results/simulation_results.xlsx")
        elif format == 'json':
            if self.baseline_results is not None:
                self.baseline_results.to_json('results/baseline_results.json', orient='records')
                print("Baseline výsledky exportované do results/baseline_results.json")
            if self.optimized_results is not None:
                self.optimized_results.to_json('results/optimized_results.json', orient='records')
                print("Optimalizované výsledky exportované do results/optimized_results.json")
        else:
            print(f"Nepodporovaný formát: {format}. Použite 'csv', 'excel' alebo 'json'.")

    def analyze_idm_market_usage(self):
        """
        Analýza využitia rôznych typov IDM trhov

        Výstup:
        -------
        dict, dict
            Počty vozidiel používajúcich rôzne typy IDM, počty slotov podľa typu IDM
        """
        # Inicializácia počítadiel
        vehicle_counts = {'15min': 0, '60min': 0, 'both': 0, 'none': 0}
        slot_counts = {'15min': 0, '60min': 0}

        # Analýza nabíjacích plánov všetkých vozidiel
        for vehicle in self.vehicles:
            used_15min = False
            used_60min = False

            for time, schedule in vehicle['charging_schedule'].items():
                if isinstance(schedule, dict) and 'source' in schedule:
                    if schedule['source'] in ['IDM', 'MIXED'] and 'idm_type' in schedule:
                        idm_type = schedule['idm_type']
                        slot_counts[idm_type] = slot_counts.get(idm_type, 0) + 1

                        if idm_type == '15min':
                            used_15min = True
                        elif idm_type == '60min':
                            used_60min = True

            # Kategorizácia vozidla
            if used_15min and used_60min:
                vehicle_counts['both'] += 1
            elif used_15min:
                vehicle_counts['15min'] += 1
            elif used_60min:
                vehicle_counts['60min'] += 1
            else:
                vehicle_counts['none'] += 1

        # Výpis výsledkov
        print("Analýza využitia IDM trhov:")
        print(f"Vozidlá používajúce iba 15-min IDM: {vehicle_counts['15min']}")
        print(f"Vozidlá používajúce iba 60-min IDM: {vehicle_counts['60min']}")
        print(f"Vozidlá používajúce oba typy IDM: {vehicle_counts['both']}")
        print(f"Vozidlá nepoužívajúce IDM: {vehicle_counts['none']}")
        print(f"Celkový počet 15-min IDM slotov: {slot_counts.get('15min', 0)}")
        print(f"Celkový počet 60-min IDM slotov: {slot_counts.get('60min', 0)}")

        return vehicle_counts, slot_counts

    def detect_price_patterns(self, prices_df, window_size=24):
        """
        Detekcia období s vysokými/nízkymi cenami

        Parametre:
        -----------
        prices_df : DataFrame
            DataFrame s cenovými údajmi
        window_size : int
            Veľkosť okna pre kĺzavý priemer (v hodinách)

        Výstup:
        -------
        dict
            Slovník s identifikovanými obdobiami
        """
        # Výpočet kĺzavého priemeru a štandardnej odchýlky
        rolling_mean = prices_df['cena'].rolling(window=window_size).mean()
        rolling_std = prices_df['cena'].rolling(window=window_size).std()

        # Identifikácia období s výrazne vysokými/nízkymi cenami
        high_price_periods = prices_df[prices_df['cena'] > (rolling_mean + rolling_std)]
        low_price_periods = prices_df[prices_df['cena'] < (rolling_mean - rolling_std)]

        # Analýza distribúcie vysokých/nízkych cien podľa hodín
        high_price_hours = high_price_periods['Time'].dt.hour.value_counts().sort_index()
        low_price_hours = low_price_periods['Time'].dt.hour.value_counts().sort_index()

        # Zobrazenie výsledkov
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        sns.barplot(x=high_price_hours.index, y=high_price_hours.values)
        plt.title('Hodiny s vysokými cenami')
        plt.xlabel('Hodina dňa')
        plt.ylabel('Počet výskytov')
        plt.xticks(range(24))

        plt.subplot(2, 1, 2)
        sns.barplot(x=low_price_hours.index, y=low_price_hours.values)
        plt.title('Hodiny s nízkymi cenami')
        plt.xlabel('Hodina dňa')
        plt.ylabel('Počet výskytov')
        plt.xticks(range(24))

        plt.tight_layout()
        plt.savefig('plots/price_patterns.png', dpi=300)
        plt.show()

        return {
            'high_price_periods': high_price_periods,
            'low_price_periods': low_price_periods,
            'high_price_hours': high_price_hours,
            'low_price_hours': low_price_hours
        }


    def calculate_average_price(self, energy_kwh, cost_eur):
        """
        Výpočet priemernej ceny energie v €/MWh

        Parametre:
        -----------
        energy_kwh : float
            Energia v kWh
        cost_eur : float
            Náklady v €

        Výstup:
        -------
        float
            Priemerná cena v €/MWh
        """
        if energy_kwh > 0:
            return (cost_eur / (energy_kwh / 1000))  # Konverzia na MWh
        return 0.0

def create_output_dirs():
    """Vytvorenie výstupných adresárov, ak neexistujú"""
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def main(FLEET_SIZE,SIMULATION_DAYS,SIMULATION_START,WORKDAYS_ONLY,DAM_ALLOCATION):

    """Hlavná funkcia pre spustenie optimalizácie nabíjania flotily EV"""
    # Vytvorenie výstupných adresárov
    create_output_dirs()

    # Nastavenie fixnej seed hodnoty pre reprodukovateľnosť
    random.seed(42)

    # Nastavenie parametra len pre pracovné dni
    workdays_only = WORKDAYS_ONLY  # Zmeniť na False pre povolenie pohybu vozidiel aj počas víkendov

    # Inicializácia optimalizátora s parametrami flotily
    fleet_size = FLEET_SIZE  # Počet vozidiel vo flotile
    optimizer = EVFleetOptimizer(
        fleet_size=fleet_size,
        ev_capacity=47,  # kWh
        min_charge_kw=1.5,
        max_charge_kw=11,
        min_soc_target=0.9,  # 90%
        workdays_only=workdays_only  # len pracovné dni
    )

    # Nastavenie cesty k údajom
    data_dir = "data"

    # Načítanie trhových údajov
    optimizer.load_market_data(data_dir)

    # Nastavenie vlastných PMF
    # PMF pre príchody (večerné hodiny, vyššie v skorom večeri)
    arrival_half_hour_probs = {
        # Polnoc až 4:00 - minimálne príchody
        (0, 0): 0, (0, 30): 0,
        (1, 0): 0, (1, 30): 0,
        (2, 0): 0, (2, 30): 0,
        (3, 0): 0, (3, 30): 0,
        (4, 0): 0, (4, 30): 0,

        # Ranné hodiny 5:00 - 9:00 - nízke príchody
        (5, 0): 0, (5, 30): 0,
        (6, 0): 0, (6, 30): 0,
        (7, 0): 0, (7, 30): 0,
        (8, 0): 0, (8, 30): 0,
        (9, 0): 0, (9, 30): 0,

        # Doobeda 10:00 - 11:30 - minimálne príchody
        (10, 0): 0, (10, 30): 0,
        (11, 0): 0, (11, 30): 0,

        # Obed 12:00 - 13:30 - nízke príchody
        (12, 0): 0, (12, 30): 0,
        (13, 0): 0, (13, 30): 0,

        # Poobede 14:00 - 15:30 - stredné príchody
        (14, 0): 0, (14, 30): 0,
        (15, 0): 0, (15, 30): 0,

        # Podvečer 16:00 - 17:30 - vyššie príchody
        (16, 0): 0, (16, 30): 5,
        (17, 0): 13, (17, 30): 14,

        # Vrchol príchodov 18:00 - 19:30
        (18, 0): 18, (18, 30): 17,
        (19, 0): 15, (19, 30): 10,

        # Večer 20:00 - 21:30 - klesajúce príchody
        (20, 0): 5, (20, 30): 3,
        (21, 0): 0, (21, 30): 0,

        # Neskorý večer 22:00 - 23:30 - nízke príchody
        (22, 0): 0, (22, 30): 0,
        (23, 0): 0, (23, 30): 0
    }

    # PMF pre odchody (ranné hodiny, vyššie počas dopravnej špičky)
    departure_half_hour_probs = {
        # Polnoc až 4:00 - minimálne odchody
        (0, 0): 0, (0, 30): 0,
        (1, 0): 0, (1, 30): 0,
        (2, 0): 0, (2, 30): 0,
        (3, 0): 0, (3, 30): 0,
        (4, 0): 0, (4, 30): 0,

        # Skoré ráno 5:00 - 5:30
        (5, 0): 0, (5, 30): 0,

        # Ranná špička 6:00 - 8:30 - vysoké odchody
        (6, 0): 15, (6, 30): 25,
        (7, 0): 35, (7, 30): 20,
        (8, 0): 3, (8, 30): 2,

        # Neskoré ráno 9:00 - 10:30 - klesajúce odchody
        (9, 0): 0, (9, 30): 0,
        (10, 0): 0, (10, 30): 0,

        # Poludnie 11:00 - 11:30 - minimálne odchody
        (11, 0): 0, (11, 30): 0,

        # Zvyšok dňa - minimálne odchody
        (12, 0): 0, (12, 30): 0,
        (13, 0): 0, (13, 30): 0,
        (14, 0): 0, (14, 30): 0,
        (15, 0): 0, (15, 30): 0,
        (16, 0): 0, (16, 30): 0,
        (17, 0): 0, (17, 30): 0,
        (18, 0): 0, (18, 30): 0,
        (19, 0): 0, (19, 30): 0,
        (20, 0): 0, (20, 30): 0,
        (21, 0): 0, (21, 30): 0,
        (22, 0): 0, (22, 30): 0,
        (23, 0): 0, (23, 30): 0
    }

    # PMF rozsahov SOC
    soc_ranges_probs = {
        (0.10, 0.15): 5,  # 10% vozidiel sa vráti s 10-20% SOC
        (0.15, 0.20): 20,  # 10% vozidiel sa vráti s 10-20% SOC
        (0.20, 0.25): 30,  # 20% vozidiel sa vráti s 20-30% SOC
        (0.25, 0.30): 20,  # 30% vozidiel sa vráti s 30-40% SOC
        (0.30, 0.35): 10,  # 25% vozidiel sa vráti s 40-50% SOC
        (0.35, 0.40): 10,  # 15% vozidiel sa vráti s 50-60% SOC
        (0.40, 0.45): 5  # 15% vozidiel sa vráti s 50-60% SOC
    }

    # Nastavenie vlastných PMF
    optimizer.set_manual_pmfs(arrival_half_hour_probs, departure_half_hour_probs, soc_ranges_probs)

    # Nastavenie parametrov simulácie
    simulation_start = SIMULATION_START
    simulation_days = SIMULATION_DAYS # Počet dní na simuláciu
    simulation_end = simulation_start + timedelta(days=simulation_days)

    # Nastavenie optimalizačných možností
    optimization_options = {
        'analyze_price_patterns': False,
        'general_idm_discount': 1,
        'targeted_night_discount': 1
    }

    # Výpis konfigurácie
    print("\n" + "=" * 60)
    print("KONFIGURÁCIA SIMULÁCIE")
    print("=" * 60)
    print(f"Veľkosť flotily: {fleet_size} vozidiel")
    print(f"Simulačné obdobie: {simulation_days} dní ({simulation_start} až {simulation_end})")
    print(f"Optimalizačné možnosti:")
    for option, value in optimization_options.items():
        print(f"  - {option}: {value}")
    print("=" * 60 + "\n")

    # Inicializácia flotily s náhodnými parametrami pre simulačné obdobie
    optimizer.initialize_fleet(simulation_start, simulation_days)

    # Spustenie základnej stratégie
    print("Spúšťanie základnej stratégie...")
    baseline_results = optimizer.baseline_strategy(simulation_start, simulation_end)
    print(f"Základná stratégia dokončená. Výsledky: {'K dispozícii' if baseline_results is not None else 'Žiadne'}")

    # Spustenie optimalizovanej stratégie s možnosťami
    print("\nSpúšťanie optimalizovanej stratégie...")
    optimized_results = optimizer.optimized_strategy(
        simulation_start,
        simulation_end,
        options=optimization_options,DAM_ALLOCATION=DAM_ALLOCATION
    )
    print(
        f"Optimalizovaná stratégia dokončená. Výsledky: {'K dispozícii' if optimized_results is not None else 'Žiadne'}")

    # Výpis základných štatistík o výsledkoch
    print("\nVýsledky:")
    print(f"Základná stratégia: {'K dispozícii' if optimizer.baseline_results is not None else 'Žiadne'}")
    print(f"Optimalizovaná stratégia: {'K dispozícii' if optimizer.optimized_results is not None else 'Žiadne'}")

    # Analýza využitia IDM trhov
    optimizer.analyze_idm_market_usage()

    # Analýza cenových vzorov
    optimizer.detect_price_patterns(optimizer.dam_prices)

    # Vizualizácia výsledkov
    optimizer.visualize_results()

    # Tlač súhrnu
    optimizer.summarize_results()

    # Export výsledkov
    optimizer.export_results(format='excel')

if __name__ == "__main__":
    # Spustenie buď hlavnej funkcie alebo porovnania
    main(FLEET_SIZE,SIMULATION_DAYS,SIMULATION_START,WORKDAYS_ONLY,DAM_ALLOCATION)  # Štandardné spustenie jednej simulácie


