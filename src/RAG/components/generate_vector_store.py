#%%
from tqdm import tqdm
import asyncio
from langchain_community.vectorstores import Chroma

#%%
def create_vector_store(collection_name, path, documents, embeddings):
    """
        The following function generates a Chroma vector store.

        Args:
            collection_name (str): The name of the collection to save the document embeddings to.
            path (str): The persist directory to save the vector store to.
            documents (List([Document])): A list of documents to add to the vector store.
            embeddings (HuggingFaceEmbeddings): The embedding model to use to generate the docunent embeddings.

        Returns:
            None: The function does not return a value. The vector store is saved at the given filepath.

        """

    vector_store = Chroma(embedding_function=embeddings, persist_directory=path,
                          collection_name=collection_name)
    print(f'VECTOR STORE INITIALIZED AT: {path}')

    async def add_documents_to_vector_store(vector_store, documents, batch_size=1000):
        # Split documents into batches

        total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size > 0 else 0)

        for i in tqdm(range(total_batches), desc="Uploading documents"):
            batch = documents[i * batch_size:(i + 1) * batch_size]
            await vector_store.aadd_documents(batch)

    asyncio.run(add_documents_to_vector_store(vector_store, documents, batch_size=1000))
    print(f'DOCUMENTS ADDED TO {collection_name} VECTOR STORE AT {path}')