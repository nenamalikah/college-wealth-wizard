#%%
from langchain_huggingface import HuggingFaceEmbeddings

import sys
sys.path.append('../../')
from components.preprocess.generate_documents import load_documents
from components.retrieval.generate_vector_store import create_vector_store
#%%
# Load IPEDs and BLS Document Objects
ipeds_docs = load_documents(document_obj_fp='../../../data/document_objs/ipeds_doc_obj.pkl')
bls_docs = load_documents(document_obj_fp='../../../data/document_objs/bls_doc_obj.pkl')
xwalk_docs = load_documents(document_obj_fp='../../../data/document_objs/crosswalk_obj.pkl')

#%%
# Select HuggingFace Model for Embeddings
embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')

#%%
# Generate IPEDs Collection in Vector Store
create_vector_store(collection_name='IPEDS_Education_Information',
                    path='../../../data/vector_store/data_store',
                    documents=ipeds_docs,
                    embeddings=embeddings)

# %%
# Generate BLS Collection in Vector Store
create_vector_store(collection_name='BLS_Occupational_Information',
                    path='../../../data/vector_store/data_store',
                    documents=bls_docs,
                    embeddings=embeddings)

#%%
# Generate XWalk Collection in Vector Store
create_vector_store(collection_name='CIP_SOC_Associations',
                    path='../../../data/vector_store/data_store',
                    documents=xwalk_docs,
                    embeddings=embeddings)
