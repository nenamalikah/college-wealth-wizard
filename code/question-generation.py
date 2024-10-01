#%%
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response


reader = SimpleDirectoryReader(input_files=["./data/CIPCode_Descriptions.csv"])
documents = reader.load_data()

#%%
import openai
openai.api_key = "*****"
data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()
eval_questions