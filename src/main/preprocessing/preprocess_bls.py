#%%
import pandas as pd
import sys

sys.path.append('../../')
from components.RAG.preprocess import clean_cip_soc_code

#  AVERAGE LENGTH OF TEXT IN BLS: 524.8323391193553
#%%
# Read bls_data.csv
bls = pd.read_csv('../../../data/bls_data.csv')

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

print(bls.columns)

#%%
# Separate the data into US-level and State-level dataframes
bls_us = bls[bls['PRIM_STATE']=='US']
bls_other = bls[bls['PRIM_STATE']!='US']

#%%
# Filter the US-level dataframe to only include the most detailed industry levels
bls_us = bls_us[bls_us['I_GROUP'].isin(['4-digit', '6-digit','cross-industry','cross-industry, ownership','4-digit, ownership', 'sector'])]
bls_us

# Check for duplicates in the US-level dataframe
dups_us = bls_us[bls_us.duplicated(subset=['NAICS','SOC_Title','Sector'],keep=False)]
dups_us

#%%
# Filter the state-level dataframe so that metropolitan and non-metropolitan areas are used
bls_other = bls_other[bls_other['AREA_TYPE'].isin([4,6])] # 185,951

# Check the State-Level dataframe for duplicates
dups_other = bls_other[bls_other.duplicated(subset=['NAICS','SOC_Title','Sector','AREA_TITLE'],keep=False)]
dups_other.sort_values(['SOC_Title','NAICS_Industry_Title'],ascending=False,inplace=True)
dups_other

#%%


bls_text = []
for idx in bls_other.index:
    bls_text.append(f'In the "{bls_other["NAICS_Industry_Title"][idx]}" industry and "{bls_other["Sector"][idx]}" sector, the mean hourly wage for a "{bls_other["SOC_Title"][idx]}" job in {bls_other["AREA_TITLE"][idx]} is {bls_other['Mean_Hourly_Wage'][idx]} and the mean annual wage is {bls_other['Mean_Annual_Wage'][idx]}. The total employment for the "{bls_other["SOC_Title"][idx]}" job in the "{bls_other["NAICS_Industry_Title"][idx]}" industry and "{bls_other["Sector"][idx]}" sector in {bls_other["AREA_TITLE"][idx]} is {bls_other["Total_Employment"][idx]}. The SOC code for the "{bls_other["SOC_Title"][idx]}" occupation is {bls_other["SOC_Code"][idx]}.')

bls_other['Occupation_Summary'] = bls_text

#%%
bls_text = []
for idx in bls_us.index:
    bls_text.append(f'In the "{bls_us["NAICS_Industry_Title"][idx]}" industry and "{bls_us["Sector"][idx]}" sector, the national mean hourly wage for a "{bls_us["SOC_Title"][idx]}" job is {bls_us['Mean_Hourly_Wage'][idx]} and the national mean annual wage is {bls_us['Mean_Annual_Wage'][idx]}. The total employment for the "{bls_us["SOC_Title"][idx]}" job in the "{bls_us["NAICS_Industry_Title"][idx]}" industry and "{bls_us["Sector"][idx]}" sector nationwide is {bls_us["Total_Employment"][idx]}. The SOC code for the "{bls_us["SOC_Title"][idx]}" occupation is {bls_us["SOC_Code"][idx]}.')

bls_us['Occupation_Summary'] = bls_text
#%%
bls = pd.concat([bls_us,bls_other],ignore_index=True)
print(f'BLS Shape {bls.shape}')
print(f'Sample BLS text: {bls_text[-1]}')


total_length = sum(len(text) for text in bls['Occupation_Summary'])
average = total_length / len(bls['Occupation_Summary'])
print(f'\n AVERAGE LENGTH OF TEXT IN BLS: {average}\n')

# bls.to_csv('../../../data/soc_employment_information.csv',index=False)