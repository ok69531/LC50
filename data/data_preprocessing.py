import random
import openpyxl
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import column 

from tqdm import tqdm

pd.set_option('mode.chained_assignment', None)
warnings.filterwarnings("ignore")


path = 'C:/Users/SOYOUNG/Desktop/toxic/data/'
chems_with_smiles = pd.read_excel(path + 'chems_with_smiles.xlsx', header = 0)
lc50 = pd.read_excel('inhalation403_lc50_re.xlsx', sheet_name = 'Sheet1')


data = pd.merge(lc50, chems_with_smiles[['CasRN', 'SMILES']], on = 'CasRN', how = 'left').reset_index(drop = True)

len(data['CasRN'].unique())


data['unit'].unique()
data['unit'].isna().sum()
data = data[data['unit'].notna()]
data = data[data['lower_value'].notna()]



casrn_na_idx = data[data['CasRN'] == '-'].index
didnt_work_idx = data[data['SMILES'] == 'Did not work'].index
data = data.drop(list(casrn_na_idx) + list(didnt_work_idx)).reset_index(drop = True)


data['value'] = [data['lower_value'][i]*0.001 if data['unit'][i] == 'mg/m^3' else data['lower_value'][i] for i in data.index]
data = data.groupby(['CasRN', 'SMILES'])['value'].mean().reset_index()

data.to_excel('lc50.xlsx', header = True, index = False)
