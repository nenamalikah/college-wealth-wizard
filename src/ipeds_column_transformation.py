#%%
import pandas as pd

df = pd.read_csv('../data/final_p1.csv')

#%%
# Categorize the variables
tuition_vars = []
housing_vars = []
expenses_vars = []
aid_vars = []
application_fee_vars = []

for col in df.columns:
    if 'tuition' in col.lower():
        tuition_vars.append(col)
    elif 'housing' in col.lower() or 'room' in col.lower() or 'board' in col.lower():
        housing_vars.append(col)
    elif 'expenses' in col.lower() or 'books and supplies' in col.lower():
        expenses_vars.append(col)
    elif 'grant' in col.lower() or 'loan' in col.lower() or 'aid' in col.lower():
        aid_vars.append(col)
    elif 'application fee' in col.lower():
        application_fee_vars.append(col)

# Output the categorized lists
print("Tuition Variables:")
print(tuition_vars)
print("\nHousing Variables:")
print(housing_vars)
print("\nExpenses Variables:")
print(expenses_vars)
print("\nAid Variables:")
print(aid_vars)
print("\nApplication Fee Variables:")
print(application_fee_vars)

#%%
for col in tuition_vars:
    df[col] = f'The {col} is ' + df[col].astype(str)
df['Tuition_Information'] = df[tuition_vars].agg(' '.join, axis=1)

df.drop(columns=tuition_vars, inplace=True)

print('Tuition Information added.')
#%%
for col in housing_vars:
    df[col] = f'The {col} is ' + df[col].astype(str)
df['Housing_Information'] = df[housing_vars].agg(' '.join, axis=1)

df.drop(columns=housing_vars, inplace=True)

print('Housing Information added.')

#%%
for col in expenses_vars:
    df[col] = f'The {col} is ' + df[col].astype(str)
df['Expenses_Information'] = df[expenses_vars].agg(' '.join, axis=1)

df.drop(columns=expenses_vars, inplace=True)

print('Expenses Information added.')

#%%
for col in aid_vars:
    df[col] = f'The {col} is ' + df[col].astype(str)
df['Aid_Information'] = df[aid_vars].agg(' '.join, axis=1)

df.drop(columns=aid_vars, inplace=True)

print('Aid Information added.')

#%%

for col in application_fee_vars:
    df[col] = f'The {col} is ' + df[col].astype(str)
df['Application_Fee_Information'] = df[application_fee_vars].agg(' '.join, axis=1)

df.drop(columns=application_fee_vars, inplace=True)

print('Application Fees Information added.')

#%%
df.drop(df.columns[0], axis=1,inplace=True)
df.drop(columns=['Institution Name'],inplace=True)
df.to_csv('../data/ipeds_vstore.csv',index=False)