from typing import Union, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain_community.llms.octoai_endpoint import OctoAIEndpoint

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ChatMessage

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

import psycopg2

from octoai.client import OctoAI
from octoai.text_gen import ChatCompletionResponseFormat
from pydantic import BaseModel

OCTOAI_API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiJkNTY1Y2Q3YS0zYmNjLTQzNDgtOGQxYy1mMGY0ZjY0ODkyYzciLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiJkMzRmOGVkZC1kMzE1LTQ4NTktOTc0Zi03MjJiMzNlNDA5ZDIiLCJ1c2VySWQiOiJhZTMxM2ExZS1lYTI3LTRkOTItYjk5OS1iOTNlY2Q0YjQ3NTgiLCJhcHBsaWNhdGlvbklkIjoiYTkyNmZlYmQtMjFlYS00ODdiLTg1ZjUtMzQ5NDA5N2VjODMzIiwicm9sZXMiOlsiRkVUQ0gtUk9MRVMtQlktQVBJIl0sInBlcm1pc3Npb25zIjpbIkZFVENILVBFUk1JU1NJT05TLUJZLUFQSSJdLCJhdWQiOiIzZDIzMzk0OS1hMmZiLTRhYjAtYjdlYy00NmY2MjU1YzUxMGUiLCJpc3MiOiJodHRwczovL2lkZW50aXR5Lm9jdG8uYWkiLCJpYXQiOjE3MTk2Nzg3NTl9.m3RpRr3vLoGekWhvk1fXHlTBAg1fkkO2V2eMfXddjLf30taFXnAOLoJIWHsGLupliLVW3JL9_wxPRhL2cGpDH_1_PeWDUYxKk_ZY1RjL9s_yoJf7fGeEy9-VPbmvjfYXcQTa_sxE99GeqwaODud26e_O04Q9n2KiNSwdeeion6ki9ctpIbDXZ3Wit_ltumGY0axDRInfm_QHwvo92r4W-3g8jN4lskS7ZA7CT5LCRq7GzFs5E2HsZq-59RnVhxnray6nG5fyG3U67Ff0ZCTuRaDY2DqRDOgQw1zRqTYMoW8zeOfozRxlGygdRcRU3AxnUOGOi7yijetkE9I60H9qlg"

conn_params = {
    'dbname': 'postgres',
    'user': 'postgres.rhgcwisgguibvcbqclrb',
    'password': 'Nosnatef78!',
    'host': 'aws-0-us-west-1.pooler.supabase.com',
    'port': '6543'
}

try:
    conn = psycopg2.connect(**conn_params)
    print("Connection established")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit()

cur = conn.cursor()

def get_string_before_substring(input_string, substring):
    # Find the position of the first occurrence of the substring
    position = input_string.find(substring)
    
    # If the substring is found, slice the string up to that position
    if position != -1:
        return input_string[:position]
    else:
        # If the substring is not found, return the original string
        return input_string
# Generate SQL Schema for my documents
# I bench pressed 100 pounds for 3 sets of 10 reps and then I squatted 50 pounds for 5 sets of 5 reps.
def parse_into_workout_schema():
    client = OctoAI(api_key=OCTOAI_API_TOKEN)
    mes = "I bench pressed 100 pounds for 3 sets of 10 reps and then I squatted 50 pounds for 5 sets of 5 reps."
    completion = client.text_gen.create_chat_completion(
        model="mistral-7b-instruct",
        messages=[ChatMessage(role="system",
                              content="You are a helpful assistant designed to output JSON. You listen for a user's workout and respond with the structured data."),
                  ChatMessage(role="user", content=mes + ". Please record this.")],
        max_tokens=512,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        response_format=ChatCompletionResponseFormat(
            type="json_object",
            schema=Workout.model_json_schema(),
        ),
    )
    result = completion.choices[0].message.content
    print(result)
    return result


class Exercise(BaseModel):
    name: str
    sets: int
    reps: int


class Workout(BaseModel):
    day: int
    exercises: List[Exercise]


class Program(BaseModel):
    name: str
    workouts: List[Workout]


chains = {}
counter = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    ''' Run at startup
        Initialise the Client and add it to app.state
    '''
    # Set data dir here
    data_dir = "../workout_data"

    files = os.listdir(data_dir)
    file_texts = []
    for file in files:
        with open(f"{data_dir}/{file}") as f:
            file_text = f.read()
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=64,
        )
        texts = text_splitter.split_text(file_text)
        for i, chunked_text in enumerate(texts):
            file_texts.append(Document(page_content=chunked_text,
                                       metadata={"doc_title": file.split(".")[0], "chunk_num": i}))

    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(
        file_texts,
        embedding=embeddings
    )
    llm = OctoAIEndpoint(
        model="meta-llama-3-8b-instruct",
        max_tokens=1024,
        presence_penalty=0,
        temperature=0.1,
        top_p=0.9,
        octoai_api_token=OCTOAI_API_TOKEN
    )
    retriever = vector_store.as_retriever()
    template = """You are a data analyst. You will be given unstructured data texts and figure out the schema. Return the schema. Be as general as possible so the schema can be used on most data input. Return the schema as PosgreSQL CREATE TABLE query. If possible, create multiple tables and relations between them when you see fit.
    Document: {question} 
    Context: {context} 
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    chains['chain'] = chain
    yield


app = FastAPI(lifespan=lifespan)
app.counter = 0

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: str, q: Union[str, None] = None):
    if app.counter is 0:
        result = chains["chain"].invoke("Show me the schema creation script, no explation, only the raw SQL script")
        substring = "INSERT"
        sql_result = get_string_before_substring(result, substring)
        
        try:
            cur.execute(sql_result)
            conn.commit()  # Commit the transaction
            print("Table created successfully")
        except Exception as e:
            print(f"Error creating table: {e}")
            conn.rollback()  # Rollback in case of error
    else:
        result = parse_into_workout_schema()
    app.counter += 1
    return {"result": result}
