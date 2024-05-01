import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import textwrap
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext ,StorageContext
from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF




os.environ["GOOGLE_API_KEY"]="AIzaSyBSzR4P5YxBXOYpZBqpPWiJ9JK0GtaTw6Y"


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def load_and_create_document(DataPath):
    documents=SimpleDirectoryReader(DataPath)
    docs=documents.load_data()
    document = Document(text="\n\n".join([doc.text for doc in docs]))
    return document


class llmRAG:
    def __init__(self, llm, DataPath):
        self.llm = llm
        self.document = load_and_create_document(DataPath)

    def serviceContextFunction(self,llm, document):
        # Create a service context using the Gemini LLM and the model
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model=embeddings
        )

        # Assuming VectorStoreIndex is properly imported and defined
        index = VectorStoreIndex.from_documents([document], service_context=service_context)
        return index

    def setupLlmRAG(self):
        self.indexLlmDepartmentClassifierRAG = self.serviceContextFunction(self.llm, self.document)
        queryEngineLlmDepartmentClassifierRAG = self.indexLlmDepartmentClassifierRAG.as_query_engine()
        self.queryEngineLlmDepartmentClassifierRAG = queryEngineLlmDepartmentClassifierRAG

    def llmDocumentsQualityCheck(self):
        response = self.queryEngineLlmDepartmentClassifierRAG.query(
            "I want you to check the quality of the data according to The Profitable Hobby Farm - 2010 - Aubrey - Sample Business Documents"
        )
        # Initialize a dictionary to store data files for each department
        return response

    def llmRAGInvoke(self, prompt):
        response = self.queryEngineLlmDepartmentClassifierRAG.query(prompt)
        # Initialize a dictionary to store data files for each department
        return response
