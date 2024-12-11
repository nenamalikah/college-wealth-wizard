# College Wealth Wizard Code Structure

### Components Folder
- The components folder contains modules for the College Wealth Wizard. 
- The modules are divided into three groups: RAG, retrieval, and preprocess.
- The preprocess module contains functions to process IPEDs and BLS data as well as generate document objects.
- The retrieval module contains functions to generate a vector store as well as retrieve documents from vector stores.
- The RAG module contains functions to create a RAG validation set as well as prompt an LLM with RAG. 

### Main Folder
- The main folder contains the source code for the College Wealth Wizard. 
- The final application file as well as the streamlit file to deploy the application are located within this folder.
- Subfolders include preprocessing, RAG, and retrieval. 
- The preprocessing folder contains the code to process the IPEDs, BLS, and CIP-SOC data.
- The RAG folder contains the code to develop the retrieval validation set.
- The retrieval folder contains the code to generate the document objects and vector store, as well as retrieval metrics. 

### Tests Folder
- The tests folder contains code to test elements of the application. 
