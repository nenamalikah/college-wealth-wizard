#%%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load Vector Store
embeddings = HuggingFaceEmbeddings(model_name = 'BAAI/bge-large-en-v1.5')
vector_store = Chroma(embedding_function=embeddings,persist_directory='../../../data/vector_store/main',collection_name='IPEDS_Education_Information')

#%%
# Set up the RAG chain using Hugging Face model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

#%%
# Query the RAG agent

queries = [
    "What CIP programs are offered by West Virginia Northern Community College?",
    "What are the average books and supplies costs for West Virginia School of Osteopathic Medicine for the 2023-24 academic year?",
    "What is the on-campus food and housing cost for West Virginia State University in 2023-24?",
    "What are the typical housing charges for an academic year at West Virginia University?",
    "Does West Virginia University at Parkersburg provide institutionally-controlled housing?",
    "What is the published out-of-state tuition and fees for West Virginia University Hospital Departments of Rad Tech and Nutrition for 2023-24?",
    "What is the average federal grant aid awarded to full-time first-time undergraduates at West Virginia University Institute of Technology?",
    "What is the undergraduate application fee for West Virginia Wesleyan College?",
    "How much is the typical board charge for an academic year at Westchester College of Nursing & Allied Health?",
    "What are the average amounts of Pell grant aid awarded to full-time first-time undergraduates at Westchester School for Medical & Dental Assistants?"]

respones = []
for query in queries:
    response = qa_chain.invoke(query)
    respones.append(response)

import pandas as pd
df = pd.DataFrame({'Questions':queries,
                   'Respones':respones})

df.to_csv('../../../data/retrieval_results/educational_rag_metrics.csv',index=False)