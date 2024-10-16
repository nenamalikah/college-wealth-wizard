import pickle
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
#%%
def save_document(csv_fp,document_obj_fp,chunk_size,chunk_overlap):
    loader = CSVLoader(file_path=csv_fp,
                       csv_args={
                           'delimiter': ',',
                       })

    docs = []
    docs_lazy = loader.lazy_load()

    for doc in tqdm(docs_lazy):
        docs.append(doc)

    docs = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_documents(docs)

    with open(document_obj_fp, "wb") as file:
        pickle.dump(docs, file)

    print(f'Documents have been uploaded to filepath: {document_obj_fp} \n')

def load_documents(document_obj_fp):
    """Load Document objects from a file."""
    with open(document_obj_fp, 'rb') as file:
        documents = pickle.load(file)
    print(f"Documents loaded from {document_obj_fp}.")
    return documents

if __name__ == "__main__":

    # Save documents to disk
    save_document(csv_fp='../data/final_p2.csv', document_obj_fp='../data/bls_docs.pkl',chunk_size=1000,chunk_overlap=0)

    # Load documents from disk
    loaded_docs = load_documents('../data/bls_docs.pkl')

    # Display loaded documents
    for doc in loaded_docs[:1]:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata} \n")
