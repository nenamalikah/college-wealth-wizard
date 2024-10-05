#%%
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
import duckdb
from langchain_community.vectorstores import DuckDB
from langchain_openai import OpenAIEmbeddings

#%%
# Load csv files into document objects

loader = CSVLoader(file_path='./data/final_p1.csv',
    csv_args={
    'delimiter': ',',
    })

docs = []
docs_lazy = loader.lazy_load()

for doc in tqdm(docs_lazy):
    docs.append(doc)

loader_bls = CSVLoader(file_path='./data/final_p2.csv',
    csv_args={
    'delimiter': ',',
    })

docs_lazy_bls = loader_bls.lazy_load()

for doc in tqdm(docs_lazy_bls):
    docs.append(doc)

#%%
# Establish connection to duckdb database and generate OpenAI embeddings
conn  = duckdb.connect("college.db")
embeddings = OpenAIEmbeddings(api_key="*****")

#%%
# Create vector store from duckdb database and add documents
vector_store = DuckDB(connection=conn, embedding=embeddings)
vector_store.add_documents(docs[:10])
