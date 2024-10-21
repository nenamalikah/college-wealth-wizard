#%%
from components.preprocess import clean_cip_soc_code
import pandas as pd

#%%
# cip_soc_crosswalk.xlsx
# Sheet CIP-SOC: CIP2020Code, CIP2020Title, SOC2018Code, SOC2018Title
crosswalk = pd.read_excel('../data/cip_soc_crosswalk.xlsx',
                          sheet_name='CIP-SOC',
                          usecols=['CIP2020Code', 'CIP2020Title', 'SOC2018Code','SOC2018Title'],
                          converters={'CIP2020Code': str,
                                      'SOC2018Code': str})

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
#
# xwalk = []
#
# for idx in crosswalk.index:
#     xwalk.append(f'The associated SOC code for CIP code {crosswalk["CIP_Code"]} is {crosswalk["SOC_Code"]}. The associated field of study for a career in {crosswalk["SOC_Title"]} is {crosswalk["CIP_Title"]}')
#
# crosswalk['CIP_SOC_Associations'] = xwalk
# print(f'The cip_soc_crosswalk shape is {crosswalk.shape}.')
# print(crosswalk)
#crosswalk.to_csv('../data/cip_soc_crosswalk_clean.csv',index=False) #  (6097, 5)
#print(f'Data file cip_soc_crosswalk.xlsx processed.')


#CIPCode_Descriptions.csv
#CIPCode, CIPDefinition, CrossReferences

# cip_descs = pd.read_csv('../data/CIPCode_Descriptions.csv',
#                           usecols=['CIPCode', 'CIPDefinition', 'CrossReferences'],
#                           converters={'CIPCode': str})
#
# cip_descs['CIP_Code'] = clean_cip_soc_code(df=cip_descs,
#                    column='CIPCode',
#                    code_type='CIP')
#
# cip_descs.drop(columns='CIPCode',inplace=True)
# print(f'The cip definitions shape is {cip_descs.shape}.') #(2848, 3)
# print(cip_descs)
#
# cip_soc_associations = crosswalk.merge(cip_descs,how='left',on='CIP_Code') #[6097 rows x 7 columns]
#
# cip_soc_associations.to_csv('../data/cip_soc_associations.csv',index=False)
# print(f'Data file cip_soc_associations.csv processed.')

#%%
#bls_data.csv
# NAICS, NAICS_TITLE, OCC_CODE, OCC_TITLE, TOT_EMP, H_MEAN, A_MEAN, PCT_TOTAL, PCT_RPT

bls = pd.read_csv('../data/bls_data.csv',
                          usecols=['OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'H_MEAN', 'A_MEAN','NAICS_TITLE'])

bls['SOC_Code'] = clean_cip_soc_code(df=bls,
                   column='OCC_CODE',
                   code_type='SOC')

bls.drop(columns='OCC_CODE',inplace=True)

bls.rename(columns={'OCC_TITLE':'SOC_Title',
                    'NAICS_TITLE':'NAICS_Industry_Title',
                    'TOT_EMP':'Total_Employment',
                    'H_MEAN':'Mean_Hourly_Wage',
                    'A_MEAN':'Mean_Annual_Wage'}, inplace=True)

print(bls.groupby(by=['SOC_Code','SOC_Title','Total_Employment','Mean_Hourly_Wage','Mean_Annual_Wage']))

bls_merge = crosswalk.groupby(by=['SOC_Code']).agg({
    'CIP_Code': lambda x: ', '.join(x),
    'CIP_Title': lambda x: ', '.join(x)
}).reset_index()
print(f'BLS Shape {bls.shape}')
bls = bls.merge(bls_merge, on='SOC_Code', how='left')

bls.to_csv('../data/soc_employment_information.csv',index=False)



#%%
#CIP University Options
#unitid, institution name, C2023_A.CIP Code -  2020 Classification, CipTitle

cip_univ_opts = pd.read_csv('../data/cip_university_options.csv',
                            usecols=['unitid', 'institution name', 'C2023_A.CIP Code -  2020 Classification','CipTitle'])

cip_univ_opts['CIP_Code'] = clean_cip_soc_code(df=cip_univ_opts,
                   column='C2023_A.CIP Code -  2020 Classification',
                   code_type='CIP')

cip_univ_opts.drop(columns='C2023_A.CIP Code -  2020 Classification',inplace=True)

cip_univ_opts = cip_univ_opts.groupby(['unitid', 'institution name']).agg({
    'CIP_Code': lambda x: ', '.join(x),
    'CipTitle': lambda x: ', '.join(x)
}).reset_index()

cip_univ_opts.rename(columns={'unitid':'Unit_ID',
                              'institution name':'Institution_Name',
                              'CIP_Code':'CIP_Codes_Offered_by_Inst',
                              'CipTitle':'CIP_Titles_Offered_by_Inst'},
                     inplace=True)
print(f'The cip_univ_opts shape is {cip_univ_opts.shape}.')
print(cip_univ_opts)


# cip_univ_opts.to_csv('../data/cip_university_options_clean.csv',index=False) # (4010, 4)
# print(f'Data file cip_university_options.csv is processed.')
#
#%%
df = pd.read_csv('../data/ipeds_data.csv',)

# # # Categorize the variables
# # tuition_vars = []
# # housing_vars = []
# # expenses_vars = []
# # aid_vars = []
# # application_fee_vars = []
# # books_supplies_vars = []
# #
# # for col in df.columns:
# #     if 'tuition' in col.lower():
# #         tuition_vars.append(col)
# #     elif 'books and supplies' in col.lower():
# #         books_supplies_vars.append(col)
# #     elif 'housing' in col.lower() or 'room' in col.lower() or 'board' in col.lower():
# #         housing_vars.append(col)
# #     elif 'expenses' in col.lower() or 'books and supplies' in col.lower():
# #         expenses_vars.append(col)
# #     elif 'grant' in col.lower() or 'loan' in col.lower() or 'aid' in col.lower():
# #         aid_vars.append(col)
# #     elif 'application fee' in col.lower():
# #         application_fee_vars.append(col)
# #
# # # Output the categorized lists
# # print("Tuition Variables:")
# # print(tuition_vars)
# # print("\nBooks and Supplies Variables:")
# # print(books_supplies_vars)
# # print("\nHousing Variables:")
# # print(housing_vars)
# # print("\nExpenses Variables:")
# # print(expenses_vars)
# # print("\nAid Variables:")
# # print(aid_vars)
# # print("\nApplication Fee Variables:")
# # print(application_fee_vars)
# #
# # #%%
# # for col in tuition_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Tuition_Information'] = df[tuition_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=tuition_vars, inplace=True)
# #
# # print('Tuition Information added.')
# #
# # #%%
# # for col in books_supplies_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Books_Information'] = df[books_supplies_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=books_supplies_vars, inplace=True)
# #
# # print('Books and Supplies Information added.')
# # #%%
# # for col in housing_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Housing_Information'] = df[housing_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=housing_vars, inplace=True)
# #
# # print('Housing Information added.')
# #
# # #%%
# # for col in expenses_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Expenses_Information'] = df[expenses_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=expenses_vars, inplace=True)
# #
# # print('Expenses Information added.')
# #
# # #%%
# # for col in aid_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Aid_Information'] = df[aid_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=aid_vars, inplace=True)
# #
# # print('Aid Information added.')
# #
# # #%%
# #
# # for col in application_fee_vars:
# #     df[col] = f'The {col} is ' + df[col].astype(str)
# # df['Application_Fee_Information'] = df[application_fee_vars].agg(' '.join, axis=1)
# #
# # df.drop(columns=application_fee_vars, inplace=True)
# #
# # print('Application Fees Information added.')
# #
 #%%
# # df = df[['UnitID','Institution Name', 'Tuition_Information',
# #                           'Books_Information', 'Housing_Information','Expenses_Information','Aid_Information',
# #                           'Application_Fee_Information']]
df.rename(columns={'UnitID':'Unit_ID'},inplace=True)
print(df) # [5994 rows x 8 columns]
df = df.merge(cip_univ_opts,how='left',on='Unit_ID') # [5994 rows x 11 columns]

df.to_csv('../data/educational_institution_information.csv',index=False)
print(f'Data file ipeds_data.csv processed.')
