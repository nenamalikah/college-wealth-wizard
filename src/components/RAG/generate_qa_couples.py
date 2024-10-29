#%%
from tqdm import tqdm
import pandas as pd
import random
from huggingface_hub import InferenceClient
import json

def generate_rag_validation(documents, repo_id, qa_prompt, n_questions, output_fp):
    """
    The following function generates question and answer couples for RAG validation.

    Args:
        documents (List[Document]): A list of LangChain Document objects containing text data.
        repo_id (str): The repo_id for the HuggingFace LLM to be used for QA generation.
        qa_prompt (str): The prompt to be provided to the LLM to generate QA couples.
        n_questions (int): The number of questions to generate.
        output_fp (str): The filepath where the generated couples will be saved.

    Returns:
        pd.DataFrame: A dataframe that generates question and answer couples from the provided documents.

    """

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

    print(f"Generating {n_questions} QA couples...")

    outputs = []
    for sampled_context in tqdm(random.sample(documents, n_questions)):
        # Generate QA couple
        output_QA_couple = call_llm(llm_client, qa_prompt.format(context=sampled_context.page_content))
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "source_doc": sampled_context.metadata["row"],
                    "answer":answer,
                }
            )
        except:
            continue

    outputs_df = pd.DataFrame(outputs)
    print(f'Here is a sample output: \n {outputs_df.iloc[0,:]}')
    outputs_df.to_excel(output_fp, index=False)
    print(f'Validation set saved at {output_fp}')



#%%

# if __name__ == '__main__':
#     generate_rag_validation(documents=crosswalk_docs,
#                             repo_id=repo_id,
#                             qa_prompt=QA_generation_prompt,
#                             n_questions=10,
#                             output_fp='../data/retrieval_results/test_generated_questions.xlsx')



