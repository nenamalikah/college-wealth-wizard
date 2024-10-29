#%%
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import sys

sys.path.append('../../')
from components.retrieval.retrieve_documents import retrieve_documents


#%%
embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')

#%%
# Retrieve Documents for BLS-Related Questions
df = pd.read_excel('../../../data/retrieval_results/bls_qa_questions.xlsx')
bls_questions = list(df['question'])
retrieve_documents(questions=bls_questions,
                   embeddings=embeddings,
                   vector_path='../../../data/vector_store/wizardt',
                   collection_name='BLS_Occupational_Information',
                   results_path='../../../data/retrieval_results/bls_retrieval.xlsx',
                   k=5)

#%%
# Retrieve Documents for IPEDs-Related Questions
df = pd.read_excel('../../../data/retrieval_results/ipeds_qa_questions.xlsx')
ipeds_questions = list(df['question'])

retrieve_documents(questions=ipeds_questions,
                   embeddings=embeddings,
                   vector_path='../../../data/vector_store/wizardt',
                   collection_name='IPEDS_Education_Information',
                   results_path='../../../data/retrieval_results/ipeds_retrieval.xlsx',
                   k=5)

#%%
# Retrieve Documents for XWalk-Related Questions
df = pd.read_excel('../../../data/retrieval_results/xwalk_qa_questions.xlsx')
xwalk_questions = list(df['question'])

retrieve_documents(questions=xwalk_questions,
                   embeddings=embeddings,
                   vector_path='../../../data/vector_store/wizardt',
                   collection_name='CIP_SOC_Associations',
                   results_path='../../../data/retrieval_results/xwalk_retrieval.xlsx',
                   k=5)