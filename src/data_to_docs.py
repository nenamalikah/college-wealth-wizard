#%%
from RAG.components.generate_documents import doc_gen_csv, doc_gen_list
import pandas as pd

#%%
ipeds_fp = '../data/educational_institution_information.csv'
ipeds_doc_fp = '../data/document_objs/ipeds_doc_obj.pkl'

ipeds = pd.read_csv(ipeds_fp, usecols=['Institution_Summary'])

doc_gen_list(text_list=ipeds['Institution_Summary'],
             doc_source='IPEDs Data Extract',
             output='Save',
             document_obj_fp=ipeds_doc_fp)

print(f'The file {ipeds_doc_fp} has been saved to {ipeds_fp}')

#%%
bls_fp = '../data/soc_employment_information.csv'
bls_doc_fp = '../data/document_objs/bls_doc_obj.pkl'

bls = pd.read_csv(bls_fp,usecols=['Occupation_Summary'])

doc_gen_list(text_list=bls['Occupation_Summary'],
             doc_source='BLS Data Extract',
             output='Save',
             document_obj_fp=bls_doc_fp)

print(f'The file {bls_fp} has been saved to {bls_doc_fp}')

#%%
crosswalk_fp = '../data/cip_soc_xwalk.csv'
crosswalk_doc_fp = '../data/document_objs/crosswalk_obj.pkl'

xwalk = pd.read_csv(crosswalk_fp, usecols=['XWalk_Summary'])

doc_gen_list(text_list=xwalk['XWalk_Summary'],
             doc_source='CIP-SOC Associations',
             output='Save',
             document_obj_fp=crosswalk_doc_fp)

print(f'The file {crosswalk_fp} has been saved to {crosswalk_doc_fp}')
