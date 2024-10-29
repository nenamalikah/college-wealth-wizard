#%%
import pandas as pd
import sys

sys.path.append('../../')
from components.RAG.preprocess import clean_cip_soc_code

#%%
#CIP University Options
#unitid, institution name, C2023_A.CIP Code -  2020 Classification, CipTitle

cip_univ_opts = pd.read_csv('../../../data/cip_university_options.csv',
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

options_list = []
for idx in cip_univ_opts.index:
    options_list.append(f'At {cip_univ_opts["Institution_Name"][idx]}, the following CIP codes are offered: {cip_univ_opts["CIP_Codes_Offered_by_Inst"][idx]}. At {cip_univ_opts["Institution_Name"][idx]}, the following CIP titles are offered: {cip_univ_opts["CIP_Titles_Offered_by_Inst"][idx]}')

cip_univ_opts['Options_Summary'] = options_list
# print(f'The cip_univ_opts shape is {cip_univ_opts.shape}.')
# print(f'Sample university options text: {options_list[100]}')


# # cip_univ_opts.to_csv('../data/cip_university_options_clean.csv',index=False) # (4010, 4)
# # print(f'Data file cip_university_options.csv is processed.')
#
#%%
df = pd.read_csv('../../../data/ipeds_data.csv')
df = df[df.columns[:38]]
print('SCRIPT BEGINS HERE')

# %%
payment = []

for answer in list(df['Tuition payment plan (IC2023)']):
    if answer == 1:
        payment.append('is provided by the institution')
    elif answer == 0:
        payment.append('is not provided by the institution')
    elif answer == -1:
        payment.append('is not reported')
    elif answer == -2:
        payment.append('is not applicable')
    else:
        payment.append('is not specified by the institution')
print(df['Tuition payment plan (IC2023)'].dtype)
df['Tuition payment plan (IC2023)'] = payment

#%%
df.rename(columns={'Institution provide nstitutionally-controlled housing (on-campus and/or off-campus) (IC2023)':
                   'Institutionally-controlled housing'},inplace=True)
housing = []
for answer in df['Institutionally-controlled housing']:
    if answer == 1:
        housing.append('is provided by the institution')
    elif answer == 2:
        housing.append('is not provided by the institution')
    elif answer == -1:
        housing.append('is not reported')
    elif answer == -2:
        housing.append('is not applicable')
    else:
        housing.append('is not reported')
df['Institutionally-controlled housing'] = housing

#%%
df.rename(columns={'Any alternative tuition plans offered by institution (IC2023)':
                   'Alternative tuition plans'},inplace=True)

plan_clean = []

for answer in df['Alternative tuition plans']:
    if answer == 1:
        plan_clean.append('are provided by the institution')
    elif answer == 2:
        plan_clean.append('are not provided by the institution')
    elif answer == -1:
        plan_clean.append('are not reported')
    elif answer == -2:
        plan_clean.append('are not applicable')
    else:
        plan_clean.append('are not specified by the institution')

df['Alternative tuition plans'] = plan_clean

#%%
df.fillna('unavailable',inplace=True)
print(df.isna().sum())

#%%
# Categorize the variables
tuition_vars = []
housing_vars = []
net_price_vars = []
aid_vars = []
application_fee_vars = []
expenses_vars = []

for col in df.columns:
    if 'tuition' in col.lower():
        tuition_vars.append(col)
    elif 'net price' in col.lower():
        net_price_vars.append(col)
    elif 'food' in col.lower() or 'room' in col.lower() or 'board' in col.lower() or 'housing' in col.lower():
        housing_vars.append(col)
    elif 'expenses' in col.lower() or 'books and supplies' in col.lower():
        expenses_vars.append(col)
    elif 'grant' in col.lower() or 'loan' in col.lower() or 'aid' in col.lower() or 'loans' in col.lower():
        aid_vars.append(col)
    elif 'application fee' in col.lower():
        application_fee_vars.append(col)


# Output the categorized lists
print("Tuition Variables:")
print(tuition_vars)
print("\nExpenses Variables:")
print(expenses_vars)
print("\nHousing Variables:")
print(housing_vars)
print("\nAid Variables:")
print(aid_vars)
print("\nApplication Fee Variables:")
print(application_fee_vars)
print("\nNet Price Variables:")
print(net_price_vars)
#%%

for col in tuition_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'
df['Tuition_Information'] = df[tuition_vars].agg(' '.join, axis=1)
df.drop(columns=tuition_vars, inplace=True)

print('Tuition Information added.')

#%%

for col in expenses_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'
df['Expenses_Information'] = df[expenses_vars].agg(' '.join, axis=1)

df.drop(columns=expenses_vars, inplace=True)

print('Exapenses Information added.')
#%%
for col in housing_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'

df['Housing_Information'] = df[housing_vars].agg(' '.join, axis=1)

df.drop(columns=housing_vars, inplace=True)

print('Housing Information added.')

#%%
for col in aid_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'

df['Aid_Information'] = df[aid_vars].agg(' '.join, axis=1)
df.drop(columns=aid_vars, inplace=True)

print('Aid Information added.')

#%%
for col in application_fee_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'
df['Application_Fee_Information'] = df[application_fee_vars].agg(' '.join, axis=1)

df.drop(columns=application_fee_vars, inplace=True)

print('Application Fees Information added.')

#%%
for col in net_price_vars:
    df[col] = f'The {col.lower()} is ' + df[col].astype(str) + '.'

df['Net_Price_Information'] = df[net_price_vars].agg(' '.join, axis=1)

df.drop(columns=net_price_vars, inplace=True)

print('Net Price Information added.')

 #%%
df = df[['UnitID','Institution Name', 'Tuition_Information',
                          'Housing_Information','Expenses_Information','Aid_Information',
                          'Application_Fee_Information', 'Net_Price_Information']]
df.rename(columns={'UnitID':'Unit_ID'},inplace=True)
df = df.merge(cip_univ_opts,how='left',on='Unit_ID') # [5994 rows x 11 columns]

summary = []
for idx in df.index:
    summary.append(f'The tuition information for {df["Institution_Name"][idx]} is as follows: {df["Tuition_Information"][idx]}; The housing information for {df["Institution_Name"][idx]} is as follows: {df["Housing_Information"][idx]}; The expenses information for {df["Institution_Name"][idx]} is as follows: {df["Expenses_Information"][idx]}; The aid information for {df["Institution_Name"][idx]} is as follows: {df["Aid_Information"][idx]}; The application fee information for {df["Institution_Name"][idx]} is as follows: {df["Application_Fee_Information"][idx]}; The average net price information at {df["Institution_Name"][idx]} is as follows: {df["Net_Price_Information"][idx]}; {df['Options_Summary'][idx]}')

df['Institution_Summary'] = summary
print(f'Sample institution summary: {summary[500]}')


total_length = sum(len(text) for text in summary)
average = total_length / len(summary)
print(f'\n AVERAGE LENGTH OF TEXT IN SUMMARY: {average}\n')

# df.to_csv('../../../data/educational_institution_information.csv',index=False)
# print(f'Data file ipeds_data.csv processed.')