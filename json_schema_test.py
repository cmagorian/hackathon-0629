from langchain import LLMChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import List
import octoai
from octoai.client import OctoAI
from octoai.text_gen import ChatCompletionResponseFormat, ChatMessage
import os

load_dotenv()


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


def parse_into_workout_schema():
    OCTOAI_API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjNkMjMzOTQ5In0.eyJzdWIiOiJkNTY1Y2Q3YS0zYmNjLTQzNDgtOGQxYy1mMGY0ZjY0ODkyYzciLCJ0eXBlIjoidXNlckFjY2Vzc1Rva2VuIiwidGVuYW50SWQiOiJkMzRmOGVkZC1kMzE1LTQ4NTktOTc0Zi03MjJiMzNlNDA5ZDIiLCJ1c2VySWQiOiJhZTMxM2ExZS1lYTI3LTRkOTItYjk5OS1iOTNlY2Q0YjQ3NTgiLCJhcHBsaWNhdGlvbklkIjoiYTkyNmZlYmQtMjFlYS00ODdiLTg1ZjUtMzQ5NDA5N2VjODMzIiwicm9sZXMiOlsiRkVUQ0gtUk9MRVMtQlktQVBJIl0sInBlcm1pc3Npb25zIjpbIkZFVENILVBFUk1JU1NJT05TLUJZLUFQSSJdLCJhdWQiOiIzZDIzMzk0OS1hMmZiLTRhYjAtYjdlYy00NmY2MjU1YzUxMGUiLCJpc3MiOiJodHRwczovL2lkZW50aXR5Lm9jdG8uYWkiLCJpYXQiOjE3MTk2Nzg3NTl9.m3RpRr3vLoGekWhvk1fXHlTBAg1fkkO2V2eMfXddjLf30taFXnAOLoJIWHsGLupliLVW3JL9_wxPRhL2cGpDH_1_PeWDUYxKk_ZY1RjL9s_yoJf7fGeEy9-VPbmvjfYXcQTa_sxE99GeqwaODud26e_O04Q9n2KiNSwdeeion6ki9ctpIbDXZ3Wit_ltumGY0axDRInfm_QHwvo92r4W-3g8jN4lskS7ZA7CT5LCRq7GzFs5E2HsZq-59RnVhxnray6nG5fyG3U67Ff0ZCTuRaDY2DqRDOgQw1zRqTYMoW8zeOfozRxlGygdRcRU3AxnUOGOi7yijetkE9I60H9qlg"
    client = OctoAI(api_key=OCTOAI_API_TOKEN)
    mes = "I bench pressed 100 pounds for 3 sets of 10 reps and then I squatted 50 pounds for 5 sets of 5 reps."
    print("Message received!: " + mes)
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
    return completion.choices[0].message.content

parse_into_workout_schema()

if __name__ == "__main__":
    pass
