#%%
from langchain_huggingface import HuggingFaceEmbeddings
from components.retrieve_documents import retrieve_documents

#%%
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2")

#%%
# Retrieve Documents for BLS-Related Questions
# retrieve_documents(questions=bls_questions,
#                    embeddings=embeddings,
#                    vector_path='../data/vector_store/main_store',
#                    collection_name='BLS_Occupational_Information',
#                    results_path='../data/retrieval_results/bls_metrics.xlsx',
#                    k=5)
#
# #%%
# # Retrieve Documents for IPEDs-Related Questions
# retrieve_documents(questions=ipeds_questions,
#                    embeddings=embeddings,
#                    vector_path='../data/vector_store/main_store',
#                    collection_name='CIP_SOC_Associations',
#                    results_path='../data/retrieval_results/cip_soc_metrics.xlsx',
#                    k=5)