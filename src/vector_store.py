#%%
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import asyncio
from components.csv_document_loader import load_documents

#%%
# Load IPEDs and BLS Document Objects
ipeds_docs = load_documents(document_obj_fp='../data/ipeds_doc_obj.pkl')

#%%
# Establish connection to Chroma database and generate HuggingFace embeddings
model_kwargs = {'device':'cuda', 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2", model_kwargs=model_kwargs)
vector_store = Chroma(embedding_function=embeddings,persist_directory='../data/cip_soc_vecstore',collection_name='ipeds_data')
# del ipeds_database
# del ipeds_nces_database
print('Vector store initialized.')

#%%
async def add_documents_to_vector_store(vector_store, documents, batch_size=1000):
    # Split documents into batches

    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)

    for i in tqdm(range(total_batches), desc="Uploading documents"):
        batch = documents[i * batch_size:(i + 1) * batch_size]
        await vector_store.aadd_documents(batch)

# Main entry point to run the asynchronous function
if __name__ == "__main__":

    asyncio.run(add_documents_to_vector_store(vector_store, ipeds_docs, batch_size=1000))
    print('BLS documents uploaded.')



