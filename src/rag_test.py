#%%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load Vector Store
model_kwargs = {'device':'cuda', 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L12-v2", model_kwargs=model_kwargs)
vector_store = Chroma(embedding_function=embeddings,persist_directory='../data/cip_soc_vecstore',collection_name='cip_soc_database')

#%%
# Set up the RAG chain using Hugging Face model
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128
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
query = "What is the average hourly wage of a data scientist"
response = qa_chain.invoke(query)
print(response)