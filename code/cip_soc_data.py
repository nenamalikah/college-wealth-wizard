#%%
import pandas as pd

# CIP Code to SOC: One to many

df = pd.read_csv('../data/cip_university_options.csv')
print(df.head())

#%%


cip_codes = df['C2023_A.CIP Code -  2020 Classification'].apply(lambda x: (''.join(filter(str.isdigit, x))))
df['CIP_Code'] = cip_codes

#%%
soc_crosswalk = pd.read_excel('data/cip_soc_crosswalk.xlsx', sheet_name='CIP-SOC', converters={'CIP2020Code':str, 'SOC2018Code':str})
print(soc_crosswalk.head())

#%%
crosswalk_cip_codes = soc_crosswalk['CIP2020Code'].apply(lambda x: (''.join(filter(str.isdigit, x))))

crosswalk_soc_codes = soc_crosswalk['SOC2018Code'].apply(lambda x: (''.join(filter(str.isdigit, x))))
soc_crosswalk['CIP_Code'] = crosswalk_cip_codes
soc_crosswalk['SOC_Code'] = crosswalk_soc_codes

print(soc_crosswalk.head())


#%%
cip_soc_data = df.merge(soc_crosswalk,how='left',on='CIP_Code')

cip_soc_data = cip_soc_data[['unitid', 'institution name', 'year', 'CIP2020Title', 'CIP_Code', 'SOC2018Title', 'SOC_Code']]
print(cip_soc_data.head())

#%%

cip_desc = pd.read_csv('../data/CIPCode_Descriptions.csv')
cip_desc.head()


cip_desc_codes = cip_desc['CIPCode'].apply(lambda x: (''.join(filter(str.isdigit, x))))
cip_desc['CIP_Code'] = cip_desc_codes

#%%

data = cip_soc_data.merge(cip_desc[['CIP_Code', 'CIPDefinition','CrossReferences']],how='left',on='CIP_Code')
data.head()

#%%

ipeds = pd.read_csv('../data/ipeds_data.csv')
ipeds.head()

ipeds = ipeds.iloc[:,0:102]

#%%
data.rename(columns={'unitid':'UnitID'},inplace=True)

#%%
data.merge(ipeds,how='inner',on='UnitID') # 627,893

#%%
data.merge(ipeds,how='left',on='UnitID')