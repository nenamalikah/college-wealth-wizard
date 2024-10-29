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

### Date: october 22 2024 
- Topics of discussion
  - RAG retrieval metrics

- Notes
  - In addition to generating accuracy metrics for RAG retrievals, we also discussed providing wrong answers to the LLM and ensuring it generates answers from the vector stores and not itself

- Action Items:

* [x] Create validation datasets for IPEDs, BLS, and CIP_SOC crosswalk files
* [ ] Generate accuracy metrics of RAG with validation datasets
* [ ] Test that generated answers from LLM are from vector stores and not LLM itself
* [ ] Read about RAG for Related Work in Report
* [ ] Clean up scripts and add testing scripts for every module (retrieval, LLM, and embeddings)

### Date: october 29 2024 
- Topics of discussion
  - RAG retrieval metrics
  - Test scripts

- Notes
  - Initial accuracy metrics were 18% for BLS dataset, 64% for IPEDs dataset, and 33% for CIP-SOC association dataset when retrieving 5 documents per query
  - When retrieving 10 documents to query, the respective metrics were 19% for BLS, 64% for IPEDs, and 41%  for XWalk
  - I analyzed the results and figured out the BLS data was duplicated on several columns that were dropped. I reprocessed this data as well as the CIP-SOC association dataset. 
  - After calculating the average length of each document, changing the chunking size, and changing the embedding model to 'BAAI/bge-large-en-v1.5', I received accuracy scores of 100, 100, 96 on random samples of 25 from the BLS, IPEDs, and CIP-SOC dataset respectively. 
  - I added a test script for calculating the accuracy of retrieval from the vector store 

- Action Items:

* [x] Create validation datasets for IPEDs, BLS, and CIP_SOC crosswalk files
* [x] Generate accuracy metrics of RAG with validation datasets
* [ ] Test that generated answers from LLM are from vector stores and not LLM itself
* [ ] Read about RAG for Related Work in Report
* [ ] Clean up scripts and add testing scripts for every module (retrieval, LLM, and embeddings)