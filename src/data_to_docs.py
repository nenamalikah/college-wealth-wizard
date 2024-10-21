#%%
from components.csv_document_loader import save_document

#%%
ipeds_fp = '../data/educational_institution_information.csv'
ipeds_doc_fp = '../data/document_objs/ipeds_doc_obj.pkl'


save_document(csv_fp=ipeds_fp,
              document_obj_fp=ipeds_doc_fp,
              chunk_size=500,
              chunk_overlap=0,
              splitter='CharacterTextSplitter')

print(f'The file {ipeds_doc_fp} has been saved to {ipeds_fp}')

#%%
bls_fp = '../data/soc_employment_information.csv'
bls_doc_fp = '../data/document_objs/bls_doc_obj.pkl'

save_document(csv_fp=bls_fp,
              document_obj_fp=bls_doc_fp,
              chunk_size=1000,
              chunk_overlap=0,
              splitter='CharacterTextSplitter')

print(f'The file {bls_fp} has been saved to {bls_doc_fp}')

# #%%
# crosswalk_fp = '../data/cip_soc_associations.csv'
# crosswalk_doc_fp = '../data/document_objs/crosswalk_obj.pkl'
#
# save_document(csv_fp=crosswalk_fp,
#               document_obj_fp=crosswalk_doc_fp,
#               chunk_size=500,
#               chunk_overlap=100,
#               splitter='CharacterTextSplitter')
#
# print(f'The file {crosswalk_fp} has been saved to {crosswalk_doc_fp}')
