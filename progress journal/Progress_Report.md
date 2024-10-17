Note: Use Markdown Cheat Sheet if you need more functionality
https://www.markdownguide.org/cheat-sheet/
### Date: sep 24 2024 
- Topics of discussion
    - Project Report
    - Project repository in GitHub

![CWW_Diagram.png](..%2Fdata%2FCWW_Diagram.png)

- Action Items:

* [ ] Merge and finalize data for CIP-SOC database (RAG Agent)
* [ ] Create validation dataset with ChatGPT (RAG Agent)
* [ ] Generate initial accuracy metrics of RAG Agent with validation dataset
* [x] Create component level, test level, and utility level folder structure (parser, unit testing)
* [ ] Decide vector store and write code for the vector store
* [ ] Decide which LLM to use and write the code 

### Date: october 1 2024 
- Topics of discussion
    - Creating AWS instance
    - Merging and finalizing data for vector database  
  

- Notes
  - The bureau of labor statistics file is too large to merge with the additional files for the CIP-SOC database
  - An AWS instance is being set up to utilize additional computing power to merge the data sources (and support future computing needs)
  - A connection was also set up between the Pycharm project that contains the capstone files and the AWS instance
  - Began developing question-generation.py to use llamaindex to develop a RAG evaluation set


- Action Items:

* [x] Upgrade type of AWS instance 
* [x] Merge and finalize data for CIP-SOC database (RAG Agent)
* [ ] Create validation dataset with ChatGPT (RAG Agent)
* [ ] Generate initial accuracy metrics of RAG Agent with validation dataset
* [x] Decide vector store and write code for the vector store
* [ ] Decide which LLM to use and write the code 

### Date: october 8 2024 
- Topics of discussion
    - Code retrieval success rate for DuckDB (Test 100 samples)
    - Test generation after code retrieval 
    - Check embeddings from HuggingFace instead of OpenAI
    - Read about RAG for Related Work in Report
  

- Notes
  - The code for the vector store and the RAG validation dataset with a sample of the CIP-SOC data was developed
  - A larger AWS instance was developed to create the full vector store and generate a larger RAG validation dataset


- Action Items:

* [ ] Create validation dataset with ChatGPT (RAG Agent)
* [ ] Generate initial accuracy metrics of RAG Agent with validation dataset
* [ ] Decide which LLM to use and write the code 
* [x] Generate code retrieval success rate for DuckDB
* [ ] Read about RAG for Related Work in Report
* [x] Change embeddings for vector store to something from Hugging Face
* [ ] Clean up scripts and add testing scripts for every module (retrieval, LLM, and embeddings)

### Date: october 17 2024 
- Topics of discussion
  - Chroma vector store upload speed
  - RAG Chain with mistralai
  

- Notes
  - I switched the embedding model from OpenAI to HuggingFace
  - I also switched the vector store from DuckDB to Chroma 
  - I was able to get initial metrics for the Chroma vector store code retrieval and it was highly successful
  - I was able to add all of the BLS data to the vector store and create a RAG chain using mistralai
  - I have only added a sample of the IPEDS data to the vector store. The IPEDs document object is very large and takes a while to be added
    - In order to speed up the process, I eliminated unnecessary columns and combined similar columns.
    - I have also tried asynchronous batch uploading to speed up the process. To date, it still takes ~3 hours
 


- Action Items:

* [ ] Create validation dataset with ChatGPT (RAG Agent)
* [ ] Generate initial accuracy metrics of RAG Agent with validation dataset
* [x] Decide which LLM to use and write the code 
* [ ] Read about RAG for Related Work in Report
* [ ] Clean up scripts and add testing scripts for every module (retrieval, LLM, and embeddings)