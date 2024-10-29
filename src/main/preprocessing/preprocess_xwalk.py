#%%
import pandas as pd
import sys

sys.path.append('../../')
from components.preprocess.data_processing import clean_cip_soc_code

#  AVERAGE LENGTH OF TEXT IN CROSSWALK: 211.2967032967033
#%%
# cip_soc_crosswalk.xlsx
# Sheet CIP-SOC: CIP2020Code, CIP2020Title, SOC2018Code, SOC2018Title
crosswalk = pd.read_excel('../../../data/cip_soc_crosswalk.xlsx',
                          sheet_name='CIP-SOC',
                          converters={'CIP2020Code': str,
                                      'SOC2018Code': str})
crosswalk.columns

#%%
crosswalk['CIP_Code'] = clean_cip_soc_code(df=crosswalk,
                   column='CIP2020Code',
                   code_type='CIP')

crosswalk['SOC_Code'] = clean_cip_soc_code(df=crosswalk,
                   column='SOC2018Code',
                   code_type='SOC')

crosswalk.drop(columns=['CIP2020Code','SOC2018Code'],inplace=True)
crosswalk.rename(columns={'CIP2020Title':'CIP_Title',
                          'SOC2018Title':'SOC_Title'},
                 inplace=True)

#%%
# Each CIP Code is matched from 1 to 23 SOC Codes
crosswalk.groupby(by=['CIP_Code', 'CIP_Title'],as_index=False)['SOC_Code'].nunique().sort_values('SOC_Code',ascending=False)

#%%
# Each SOC Code is matched from 1 to 337 SOC Codes
crosswalk.groupby(by=['SOC_Code', 'SOC_Title'],as_index=False)['CIP_Code'].nunique().sort_values('CIP_Code',ascending=False)

#%%
no_cip_for_soc = crosswalk[crosswalk['CIP_Code']=='999999']

xwalk = []
for idx in no_cip_for_soc.index:
    xwalk.append(f'For SOC code "{no_cip_for_soc["SOC_Code"][idx]}", there is no associated CIP code. For a "{no_cip_for_soc["SOC_Title"][idx]}" career, there is no associated field of study.')

no_cip_for_soc['XWalk_Summary'] = xwalk

#%%
no_soc_for_cip = crosswalk[crosswalk['SOC_Code']=='999999']

xwalk = []
for idx in no_soc_for_cip.index:
    xwalk.append(
        f'For CIP code "{no_soc_for_cip["CIP_Code"][idx]}", there is no associated SOC code. For a field of study in "{no_soc_for_cip["CIP_Title"][idx]}", there is no associated career path.')

no_soc_for_cip['XWalk_Summary'] = xwalk

#%%
all_codes = crosswalk[(crosswalk['SOC_Code']!='999999') &
                      (crosswalk['CIP_Code']!='999999')]

xwalk = []

for idx in all_codes.index:
    xwalk.append(f'For CIP Code "{all_codes["CIP_Code"][idx]}", the associated SOC codes are as follows: {all_codes["SOC_Code"][idx]}. For the "{all_codes["CIP_Title"][idx]}" field of study, the associated career paths are as follows: {all_codes["SOC_Title"][idx]}.')

all_codes['XWalk_Summary'] = xwalk

#%%
final = pd.concat([all_codes, no_cip_for_soc, no_soc_for_cip],ignore_index=True)

print(f'The cip_soc_crosswalk shape is {final.shape}.')
print(f'Sample crosswalk text: {final['XWalk_Summary'][100]}')

total_length = sum(len(text) for text in final['XWalk_Summary'])
average = total_length / len(final['XWalk_Summary'])
print(f'\n AVERAGE LENGTH OF TEXT IN CROSSWALK: {average}\n')

# final.to_csv('../../../data/cip_soc_xwalk.csv',index=False) #  (6097, 5)
# print(f'Data file cip_soc_crosswalk.xlsx processed.')
# print(final.columns)