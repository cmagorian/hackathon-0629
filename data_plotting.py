from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


embeddings = HuggingFaceEmbeddings()

from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class Agent:
    def __init__(self, llm, file_texts, embeddings):
        vector_store = FAISS.from_documents(
        file_texts,
        embedding=embeddings
        )
        self.retriever = vector_store.as_retriever()
        self.llm = llm

    def set_template(self, template):
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
        {"context": self.retriever, "question": RunnablePassthrough()}
        | prompt
        | self.llm
        | StrOutputParser()
        )
        
    def respond(self, query):
        return self.chain.invoke(query)
    


import json
import json_schema_test

def load_workout_data(json_path):
    file_texts = []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=64
    )
    
    for i, entry in enumerate(data['workouts']):
        workout_json = json.dumps(entry, indent=4)
        texts = text_splitter.split_text(workout_json)
        
        for j, chunked_text in enumerate(texts):
            file_texts.append(Document(
                page_content=chunked_text, 
                metadata={"doc_title": f"workout_{i+1}", "chunk_num": j}
            ))
    
    return file_texts




file_texts = load_workout_data(json_data)
llm_plotter = OctoAIEndpoint(
        model="meta-llama-3-8b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        octoai_api_token=OCTOAI_API_TOKEN
    )
plotter = Agent(llm_plotter, file_texts=file_texts, embeddings=embeddings)
template="""You are a Python developer. You will be given a workout history in a json schemas, and it's your job to figure out how to best visualize the associated data. Return Python code that creates the visualization using matplotlib or seaborn from those tables. Assume the json strings are preloaded into a variable called schemas.
Document: {question}
Context: {context}
Answer:"""
plotter.set_template(template)

import re

def parse_python_code(code):
    pattern = r"```[Pp]ython(.*?)```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        code = match.group(1).strip()
        #print(code)
    else:
        print("No Python code found.")
    return code

# code = plotter.respond("Show me the Python code to visualize the data")
# print(code)
code = parse_python_code(code)
exec(code)
