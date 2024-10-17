#%%
from components.csv_document_loader import save_document

ipeds_fp = '../data/ipeds_vstore.csv'
ipeds_doc_fp = '../data/ipeds_doc_obj.pkl'

# bls_fp = '../data/final_p2.csv'
# bls_doc_fp = '../data/bls_doc_obj.pkl'

#%%
save_document(csv_fp=ipeds_fp,
              document_obj_fp=ipeds_doc_fp,
              chunk_size=1000,
              chunk_overlap=0)

print(f'The file {ipeds_doc_fp} has been saved to {ipeds_fp}')

#%%
# save_document(csv_fp=bls_fp,
#               document_obj_fp=bls_doc_fp,
#               chunk_size=1000,
#               chunk_overlap=0)
#
# print(f'The file {bls_fp} has been saved to {bls_doc_fp}')

