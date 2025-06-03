import pandas as pd
import numpy as np

# Load data
data = pd.read_pickle('all_points_may_2024-001.pkl')

# 1. Your mapping
soo_to_side = {
    'RVOTSEPTUM': 'Right',
    'RVOTFREEWALL': 'Right',
    'RCC': 'Right',
    'LVOTSUMMIT': 'Left',
    'LVOTSUBVALVULAR': 'Left',
    'LCC': 'Left',
    'COMMISURE': 'Left',
    'INTERSENOSDCHOIZDO': 'Left'
}

# 2. Load mapping Excel and build (SOO_chamber, SOO) → Simplified
labels_excel = pd.read_excel('labels_FontiersUnsupervised.xlsx', sheet_name='Hoja2')
labels_excel = labels_excel.dropna(subset=['SOO_chamber', 'SOO', 'Simplified'])

# Clean function
def clean_str(x):
    return str(x).strip().upper().replace(' ', '').replace('.', '').replace(',', '').replace('-', '')

labels_excel['SOO_key'] = labels_excel.apply(
    lambda row: (clean_str(row['SOO_chamber']), clean_str(row['SOO'])), axis=1
)
labels_excel['Simplified_clean'] = labels_excel['Simplified'].apply(clean_str)

# Dict for mapping (SOO_chamber, SOO) → Simplified
soo_mapping = dict(zip(labels_excel['SOO_key'], labels_excel['Simplified_clean']))

# 3. Build DataFrame from main data
def build_ecg_dataframe(data_dict, mapping, soo_to_side):
    rows = []
    for patient_id, patient in data_dict.items():
        raw_soo_chamber = patient.get('SOO_chamber', '')
        raw_soo = patient.get('SOO', '')

        if isinstance(raw_soo_chamber, list):
            raw_soo_chamber = raw_soo_chamber[0] if raw_soo_chamber else ''
        if isinstance(raw_soo, list):
            raw_soo = raw_soo[0] if raw_soo else ''

        cleaned_key = (clean_str(raw_soo_chamber), clean_str(raw_soo))
        simplified = mapping.get(cleaned_key, 'Unknown')
        side = soo_to_side.get(simplified, 'Unknown')

        structures = patient.get('Structures', {})
        if not isinstance(structures, dict):
            continue

        for structure_key, px_dict in structures.items():
            if not isinstance(px_dict, dict):
                continue
            for pxx_key, leads_dict in px_dict.items():
                if not isinstance(leads_dict, dict):
                    continue

                row = {
                    'PatientID': patient_id,
                    'Structure': structure_key,
                    'PXX': pxx_key,
                    'SOO_chamber': raw_soo_chamber,
                    'SOO': raw_soo,
                    'Simplified': simplified,
                    'Side': side
                }

                for lead in ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                    row[lead] = leads_dict.get(lead, np.array([]))

                rows.append(row)

    return pd.DataFrame(rows)

# 4. Run and save
ecg_df = build_ecg_dataframe(data, soo_mapping, soo_to_side)
ecg_df = ecg_df[ecg_df['Side'] != 'Unknown'].copy()
ecg_df.to_pickle("ecg_dataset.pkl")
