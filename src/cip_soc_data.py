#%%
import pandas as pd

#%%
# CIP Code to SOC: One to many
df = pd.read_csv('./data/cip_university_options.csv')
print(df.head())

#%%
# Clean the CIP codes in the university options file
cip_codes = df['C2023_A.CIP Code -  2020 Classification'].apply(lambda x: (''.join(filter(str.isdigit, x))))
df['CIP_Code'] = cip_codes

#%%
# Load and clean the CIP and SOC codes in the crosswalk file
soc_crosswalk = pd.read_excel('./data/cip_soc_crosswalk.xlsx', sheet_name='CIP-SOC', converters={'CIP2020Code':str, 'SOC2018Code':str})

crosswalk_cip_codes = soc_crosswalk['CIP2020Code'].apply(lambda x: (''.join(filter(str.isdigit, x))))
crosswalk_soc_codes = soc_crosswalk['SOC2018Code'].apply(lambda x: (''.join(filter(str.isdigit, x))))
soc_crosswalk['CIP_Code'] = crosswalk_cip_codes
soc_crosswalk['SOC_Code'] = crosswalk_soc_codes

print(soc_crosswalk.head())


#%%
# Merge the data from the crosswalk and university options files
cip_soc_data = df.merge(soc_crosswalk,how='left',on='CIP_Code')
cip_soc_data = cip_soc_data[['unitid', 'institution name', 'year', 'CIP2020Title', 'CIP_Code', 'SOC2018Title', 'SOC_Code']]
print(cip_soc_data.head())

#%%
# Load the CIP descriptions file and clean its codes
cip_desc = pd.read_csv('./data/CIPCode_Descriptions.csv')
cip_desc_codes = cip_desc['CIPCode'].apply(lambda x: (''.join(filter(str.isdigit, x))))
cip_desc['CIP_Code'] = cip_desc_codes

#%%
# Merge the CIP descriptions with the main dataframe
cip_soc_data = cip_soc_data.merge(cip_desc[['CIP_Code', 'CIPDefinition','CrossReferences']],how='left',on='CIP_Code')
cip_soc_data.head()

#%%
# Load the ipeds data
ipeds = pd.read_csv('./data/ipeds_data.csv')
ipeds.head()
ipeds = ipeds.iloc[:,0:102]

#%%
cip_soc_data.rename(columns={'unitid':'UnitID'},inplace=True)
data = cip_soc_data.merge(ipeds,how='inner',on='UnitID')
data.sort_values('SOC_Code',inplace=True)
data.to_csv('./data/final_p1.csv')
print(data.shape)

#%%
bls = pd.read_csv('./data/bls_data.csv')
bls['SOC_Code'] = bls['OCC_CODE'].apply(lambda x: (''.join(filter(str.isdigit, x))))
bls.to_csv('./data/final_p2.csv')

#%%
bls_dask = dataframe.from_pandas(bls,npartitions=6)

#%%
final_data = dataframe.merge(left=data,right=bls_dask,on='SOC_Code',how='inner')
final_data.shape
#%%
