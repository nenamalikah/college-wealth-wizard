#%%
from langchain_huggingface import HuggingFaceEmbeddings
import random

sys.path.append('../../')
from components.preprocess.generate_documents import load_documents
from components.retrieval.generate_vector_store import create_vector_store
#%%
# Load IPEDs and BLS Document Objects

random.seed(6303)

ipeds_docs = load_documents(document_obj_fp='../../../data/document_objs/ipeds_doc_obj.pkl')
bls_docs = load_documents(document_obj_fp='../../../data/document_objs/bls_doc_obj.pkl')
xwalk_docs = load_documents(document_obj_fp='../../../data/document_objs/crosswalk_obj.pkl')

#%%
# Select HuggingFace Model for Embeddings
# model_kwargs = {'device':'cuda', 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')

#%%
# Generate IPEDs Collection in Vector Store
create_vector_store(collection_name='IPEDS_Education_Information',
                    path='../../../data/vector_store/wizardt',
                    documents=random.sample(ipeds_docs,k=100),
                    embeddings=embeddings)

# %%
# Generate IPEDs Collection in Vector Store
create_vector_store(collection_name='BLS_Occupational_Information',
                    path='../../../data/vector_store/wizardt',
                    documents=random.sample(bls_docs,k=100),
                    embeddings=embeddings)

#%%
# Generate Xwalk Collection in Vector Store
create_vector_store(collection_name='CIP_SOC_Associations',
                    path='../../../data/vector_store/wizardt',
                    documents=random.sample(xwalk_docs,k=100),
                    embeddings=embeddings)

# vector_store = Chroma(embedding_function=embeddings,persist_directory='../data/vector_store/ipeds',collection_name='ipeds_data')
#
# print('Vector store initialized.')
# #%%
# async def add_documents_to_vector_store(vector_store, documents, batch_size=1000):
#     # Split documents into batches
#
#     total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)
#
#     for i in tqdm(range(total_batches), desc="Uploading documents"):
#         batch = documents[i * batch_size:(i + 1) * batch_size]
#         await vector_store.aadd_documents(batch)
#
# # Main entry point to run the asynchronous function
# if __name__ == "__main__":
#     asyncio.run(add_documents_to_vector_store(vector_store, ipeds_docs, batch_size=1000))
#
#

