#%%
import pandas as pd
import sys

sys.path.append('../../')
from components.preprocess.generate_documents import doc_gen_list

#%%
# ipeds_fp = '../../../data/educational_institution_information.csv'
# ipeds_doc_fp = '../../../data/document_objs/ipeds_doc_obj.pkl'
#
# ipeds = pd.read_csv(ipeds_fp, usecols=['Institution_Summary'])
#
# # doc_gen_list(text_list=ipeds['Institution_Summary'],
# #              doc_source='IPEDs Data Extract',
# #              chunk_size=250,
# #              chunk_overlap=50,
# #              sep=';',
# #              output='Save',
# #              document_obj_fp=ipeds_doc_fp)
#
# doc_gen_list(text_list=ipeds['Institution_Summary'],
#              doc_source='IPEDs Data Extract',
#              chunk_size=250,
#              chunk_overlap=50,
#              sep='.',
#              output='Save',
#              document_obj_fp=ipeds_doc_fp)
#
# print(f'The file {ipeds_doc_fp} has been saved to {ipeds_fp}')
#
#%%
bls_fp = '../../../data/soc_employment_information.csv'
bls_doc_fp = '../../../data/document_objs/bls_doc_obj.pkl'

bls = pd.read_csv(bls_fp,usecols=['Occupation_Summary'])

doc_gen_list(text_list=bls['Occupation_Summary'],
             doc_source='BLS Data Extract',
             output='Save',
             chunk_size=150,
             chunk_overlap=25,
             sep=';',
             document_obj_fp=bls_doc_fp)

print(f'The file {bls_fp} has been saved to {bls_doc_fp}')
#
#%%
crosswalk_fp = '../../../data/cip_soc_xwalk.csv'
crosswalk_doc_fp = '../../../data/document_objs/crosswalk_obj.pkl'

xwalk = pd.read_csv(crosswalk_fp)
print(xwalk.columns)

text = xwalk['XWalk_Summary']
doc_gen_list(text_list=text,
             doc_source='CIP-SOC Associations',
             output='Save',
             chunk_size=100,
             chunk_overlap=25,
             document_obj_fp=crosswalk_doc_fp)

print(f'The file {crosswalk_fp} has been saved to {crosswalk_doc_fp}')
