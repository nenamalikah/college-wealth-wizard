#%%
from langchain_community.document_loaders import CSVLoader
from tqdm import tqdm
import pandas as pd
import random
from huggingface_hub import InferenceClient
import json

loader = CSVLoader(file_path='../data/final_p1.csv',
    csv_args={
    'delimiter': ',',
    })

docs = []
docs_lazy = loader.lazy_load()

for doc in tqdm(docs_lazy):
    docs.append(doc)

#%%
loader_bls = CSVLoader(file_path='../data/final_p2.csv',
    csv_args={
    'delimiter': ',',
    })

docs_lazy_bls = loader_bls.lazy_load()

for doc in tqdm(docs_lazy_bls):
    docs.append(doc)

#%%
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(
    model=repo_id,
    timeout=120,
)

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


call_llm(llm_client, "This is a test context")

#%%

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

N_GENERATIONS = 100  # We intentionally generate only 10 QA couples here for cost and time considerations

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs, N_GENERATIONS)):
    # Generate QA couple
    output_QA_couple = call_llm(llm_client, QA_generation_prompt.format(context=sampled_context.page_content))
    try:
        question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
        answer = output_QA_couple.split("Answer: ")[-1]
        assert len(answer) < 300, "Answer is too long"
        outputs.append(
            {
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata["source"],
            }
        )
    except:
        continue


print(outputs)
#%%
outputs_df = pd.DataFrame(outputs)
outputs_df.to_csv('Sample_RAG_Validation_Set.csv')