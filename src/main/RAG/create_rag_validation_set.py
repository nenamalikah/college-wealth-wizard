#%%
import random
import sys

sys.path.append('../../')
from components.RAG.generate_qa_couples import generate_rag_validation
from components.preprocess.generate_documents import load_documents

#%%
# Load Document Objects

ipeds_docs = load_documents(document_obj_fp='../../../data/document_objs/ipeds_doc_obj.pkl')
bls_docs = load_documents(document_obj_fp='../../../data/document_objs/bls_doc_obj.pkl')
crosswalk_docs = load_documents(document_obj_fp='../../../data/document_objs/crosswalk_obj.pkl')

random.seed(6303)

ipeds_docs = random.sample(ipeds_docs, 100)
bls_docs = random.sample(bls_docs, 100)
crosswalk_docs = random.sample(crosswalk_docs, 100)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

#%%
# Define the QA Generation Prompt

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

#%%
# Generate rag validation sets for each collection in the vector store

generate_rag_validation(documents=ipeds_docs,
                        repo_id=repo_id,
                        qa_prompt=QA_generation_prompt,
                        n_questions=25,
                        output_fp='../../../data/retrieval_results/ipeds_qa_questions.xlsx')

generate_rag_validation(documents=bls_docs,
                        repo_id=repo_id,
                        qa_prompt=QA_generation_prompt,
                        n_questions=25,
                        output_fp='../../../data/retrieval_results/bls_qa_questions.xlsx')

generate_rag_validation(documents=crosswalk_docs,
                        repo_id=repo_id,
                        qa_prompt=QA_generation_prompt,
                        n_questions=25,
                        output_fp='../../../data/retrieval_results/xwalk_qa_questions.xlsx')


