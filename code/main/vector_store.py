#%%
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.load import load
# import json

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
import asyncio
#from components.csv_document_loader import save_document, load_documents


#%%
# # Load IPEDs and BLS Document Objects
# with open("../data/ipeds_document_obj.json", "r") as fp:
#     ipeds_docs = load(json.load(fp))
#
# with open("../data/bls_document_obj.json", "r") as fp:
#     bls_docs = load(json.load(fp))
# print('Documents loaded.')

#%%
# Load csv files into document objects

loader = CSVLoader(file_path='../../data/final_p1.csv',
                   metadata_columns=['year','UnitID','CIP_Code','SOC_Code','CIP2020Title','SOC2018Title','institution name','CIPDefinition'],
                   csv_args={
    'delimiter': ',',
    })

docs = []
docs_lazy = loader.lazy_load()

for doc in tqdm(docs_lazy):
    docs.append(doc)

loader_bls = CSVLoader(file_path='../../data/final_p2.csv',
                       metadata_columns=['SOC_Code','OCC_TITLE','NAICS','NAICS_TITLE'],
                       csv_args={
    'delimiter': ',',
    })

docs_lazy_bls = loader_bls.lazy_load()

for doc in tqdm(docs_lazy_bls):
    docs.append(doc)

#%%
documents = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(docs[:1000])
print('Documents have been splitted.')

#%%
# Establish connection to duckdb database and generate OpenAI embeddings
model_kwargs = {'device':'cuda', 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2", model_kwargs=model_kwargs)
vector_store = Chroma(embedding_function=embeddings,persist_directory='../data/cip_soc_db_storev2',collection_name='cip_soc_database')

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
    asyncio.run(add_documents_to_vector_store(vector_store, documents, batch_size=1000))

