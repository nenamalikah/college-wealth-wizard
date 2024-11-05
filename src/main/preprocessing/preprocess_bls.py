#%%
import pandas as pd
import sys
import numpy as np
sys.path.append('../../')
from components.preprocess.data_processing import clean_cip_soc_code

#  AVERAGE LENGTH OF TEXT IN BLS: 440.9265947118954
#%%
# Read bls_data.csv
# Define the converter function
def convert_to_int_with_commas(x):
    try:
        # Remove commas and attempt to convert to integer
        return int(x.replace(',', ''))
    except ValueError:
        # Return NaN or a default value if conversion fails
        return np.nan

def convert_to_float_with_commas(x):
    try:
        # Remove commas and attempt to convert to integer
        return int(x.replace(',', ''))
    except ValueError:
        # Return NaN or a default value if conversion fails
        return np.nan

bls = pd.read_csv('../../../data/bls_data.csv',
                  converters={'TOT_EMP':convert_to_int_with_commas,
                              'H_MEAN':convert_to_float_with_commas,
                              'A_MEAN':convert_to_float_with_commas})

# Filter the data to only use detailed SOC levels
bls = bls[bls['O_GROUP']=='detailed']

#%%
# Clean the SOC code's
bls['SOC_Code'] = clean_cip_soc_code(df=bls,
                   column='OCC_CODE',
                   code_type='SOC')

bls.drop(columns='OCC_CODE',inplace=True)

bls.rename(columns={'OCC_TITLE':'SOC_Title',
                    'NAICS_TITLE':'NAICS_Industry_Title',
                    'TOT_EMP':'Total_Employment',
                    'H_MEAN':'Mean_Hourly_Wage',
                    'A_MEAN':'Mean_Annual_Wage',
                    'OWN_CODE':'Sector'}, inplace=True)

#
bls['Sector'] = bls['Sector'].replace({1:'Federal Government',
                       2:'State Government',
                       3:'Local Government',
                       4:'Federal, State, and Local Government',
                       235:'Private, State, and Local Government',
                       35: 'Private and Local Government',
                       5: 'Private',
                       57: 'Private, Local Government Gambling Establishments (Sector 71), and Local Government Casino Hotels (Sector 72)',
                       58:'Private plus State and Local Government Hospitals',
                       59: 'Private and Postal Service',
                       1235:'Federal, State, and Local Government and Private Sector'})

# bls['Total_Employment'] = bls['Total_Employment'].replace({'**':'unavailable'})

print(bls.columns)

#%%
# Separate the data into US-level and State-level dataframes
bls_us = bls[bls['PRIM_STATE']=='US']
bls_other = bls[bls['PRIM_STATE']!='US']

#%%
# Filter the US-level dataframe to only include the most detailed industry levels
bls_us = bls_us[bls_us['I_GROUP'].isin(['4-digit', '6-digit','cross-industry','cross-industry, ownership','4-digit, ownership', 'sector'])]
bls_us

#%%

bls_us = bls_us.groupby(by=['PRIM_STATE', 'NAICS_Industry_Title', 'SOC_Title','SOC_Code'],as_index=False)[['Total_Employment','Mean_Hourly_Wage', 'Mean_Annual_Wage']].mean().sort_values(by=['PRIM_STATE','SOC_Title','NAICS_Industry_Title'])
bls_us.head()
#%%
# Check for duplicates in the US-level dataframe
dups_us = bls_us[bls_us.duplicated(subset=['NAICS_Industry_Title','SOC_Title','PRIM_STATE'],keep=False)]
dups_us

#%%
# Filter the state-level dataframe so that metropolitan and non-metropolitan areas are used
bls_other = bls_other[bls_other['AREA_TYPE'] == 2] # 185,951

# Filter the US-level dataframe to only include the most detailed industry levels
bls_other = bls_other[bls_other['I_GROUP'].isin(['4-digit', '6-digit','cross-industry','cross-industry, ownership','4-digit, ownership', 'sector'])]
bls_other
#%%
bls_other = bls_other.groupby(by=['PRIM_STATE', 'NAICS_Industry_Title', 'SOC_Title','SOC_Code'],as_index=False)[['Total_Employment','Mean_Hourly_Wage', 'Mean_Annual_Wage']].mean().sort_values(by=['PRIM_STATE','SOC_Title','NAICS_Industry_Title'])
bls_other.head()
#%%
# Check the State-Level dataframe for duplicates
dups_other = bls_other[bls_other.duplicated(subset=['NAICS_Industry_Title','SOC_Title','PRIM_STATE'],keep=False)]
dups_other.sort_values(['SOC_Title','NAICS_Industry_Title'],ascending=False,inplace=True)
dups_other

#%%
bls_us.fillna('unavailable',inplace=True)
bls_other.fillna('unavailable',inplace=True)

#%%
bls_text = []
for idx in bls_other.index:
    bls_text.append(f'For the "{bls_other["SOC_Title"][idx]}" occupation located in {bls_other["PRIM_STATE"][idx]} within the "{bls_other["NAICS_Industry_Title"][idx]}" industry, the mean hourly wage is {bls_other['Mean_Hourly_Wage'][idx]} and the mean annual wage is {bls_other['Mean_Annual_Wage'][idx]}.. The total employment is {bls_other["Total_Employment"][idx]} for the "{bls_other["SOC_Title"][idx]}" occupation located in {bls_other["PRIM_STATE"][idx]} within the "{bls_other["NAICS_Industry_Title"][idx]}" industry.. The SOC code for the "{bls_other["SOC_Title"][idx]}" occupation within the "{bls_other["NAICS_Industry_Title"][idx]}" industry is {bls_other["SOC_Code"][idx]}..')

bls_other['Occupation_Summary'] = bls_text

#%%
bls_text = []
for idx in bls_us.index:
    bls_text.append(f'For the "{bls_us["SOC_Title"][idx]}" occupation within the "{bls_us["NAICS_Industry_Title"][idx]}" industry, the national mean hourly wage is {bls_us['Mean_Hourly_Wage'][idx]} and the national mean annual wage is {bls_us['Mean_Annual_Wage'][idx]}.. The total employment nationwide is {bls_us["Total_Employment"][idx]} for the "{bls_us["SOC_Title"][idx]}" occupation in the "{bls_us["NAICS_Industry_Title"][idx]}" industry.. The SOC code for the "{bls_us["SOC_Title"][idx]}" occupation within the "{bls_us["NAICS_Industry_Title"][idx]}" industry is {bls_us["SOC_Code"][idx]}..')

bls_us['Occupation_Summary'] = bls_text
#%%
bls = pd.concat([bls_us,bls_other],ignore_index=True)
print(f'BLS Shape {bls.shape}')
print(f'Sample BLS text: {bls_text[-1]}')


total_length = sum(len(text) for text in bls['Occupation_Summary'])
average = total_length / len(bls['Occupation_Summary'])
print(f'\n AVERAGE LENGTH OF TEXT IN BLS: {average}\n')

bls_us.to_csv('../../../data/soc_employment_information.csv',index=False)