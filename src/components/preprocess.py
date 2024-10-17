import pandas as pd

def clean_cip_soc_code(df, column, code_type):

    if code_type == 'CIP':
        clean_column = df[column].apply(lambda x: (''.join(filter(str.isdigit, x))))
        return clean_column

    elif code_type == 'SOC':
        clean_column = df[column].apply(lambda x: (''.join(filter(str.isdigit, x))))
        return clean_column
