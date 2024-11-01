import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id
)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a series of vector stores or a web search. For questions on tuition, financial aid, room and board, books, offered fields of study, and other collegiate expenses at colleges and universities, use the IPEDs vector store. For questions related to mean average wages, total employment, and SOC codes of specific occupations, use the BLS vector store. For questions about fields of study (CIP codes/titles) and their associated occupations (SOC codes/titles), use the CIP_SOC vector store. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give one of the following choices based on the question: 'IPEDs vector store', 'BLS vector store',  'CIP_SOC vector store', or 'web_search'. Return the a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)
start = time.time()
question_router = prompt | llm | JsonOutputParser()

question = "associated occupation for field of study in biology"
print(question_router.invoke({"question": question}))
end = time.time()
print(f"The time required to generate response by Router Chain in seconds:{end - start}")