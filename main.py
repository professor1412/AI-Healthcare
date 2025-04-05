import os
import sys
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langgraph.prebuilt import create_react_agent
from rag import create_healthcare_retriever
from templates import (
    cardiologist_template,
    psychologist_template,
    pulmonologist_template,
    multidisciplinary_prompt,
)
from langchain.tools.retriever import create_retriever_tool
from tools import tavily_tool
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Set Python recursion limit (workaround for LangChain recursion depth)
sys.setrecursionlimit(10000)

# Load environment variables and initialize ChatGroq
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Initialize FastAPI app and setup CORS middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Healthcare AI Assistant API is running."}

def process_files(files: List[UploadFile]):
    try:
        docs = []
        for uploaded_file in files:
            try:
                uploaded_file.file.seek(0)
                reader = PdfReader(uploaded_file.file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                docs.append(Document(page_content=text, metadata={"source": uploaded_file.filename}))
            except Exception as e:
                print(f"Error processing file {uploaded_file.filename}: {e}")
                raise HTTPException(status_code=400, detail=f"Error processing file {uploaded_file.filename}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error in process_files: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing files.")

def setup_retriever(pdf_content):
    try:
        retriever = create_healthcare_retriever(pdf_content)
        retrieval_tool = create_retriever_tool(
            retriever,
            "Pdf_content_retriever",
            "Searches and returns excerpts from uploaded medical PDFs.",
        )
        return retriever, retrieval_tool
    except Exception as e:
        print(f"Error setting up retriever: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while setting up retriever.")

def setup_agents(tools):
    try:
        cardiologist_graph = create_react_agent(chat, tools=tools, state_modifier=cardiologist_template)
        psychologist_graph = create_react_agent(chat, tools=tools, state_modifier=psychologist_template)
        pulmonologist_graph = create_react_agent(chat, tools=tools, state_modifier=pulmonologist_template)
        return cardiologist_graph, psychologist_graph, pulmonologist_graph
    except Exception as e:
        print(f"Error setting up agents: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while setting up agents.")

@app.post("/healthcare-assist/")
async def healthcare_assist(
    query: str = Form(...),
    option: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query is required.")

        pdf_content = process_files(files)
        retriever, retrieval_tool = setup_retriever(pdf_content)
        tools = [tavily_tool, retrieval_tool]
        cardiologist_graph, psychologist_graph, pulmonologist_graph = setup_agents(tools)

        inputs = {"messages": [("human", query)]}

        if option == "Cardiologist":
            async for chunk in cardiologist_graph.astream(inputs, stream_mode="values"):
                final_result = chunk
            return {"cardiologist_advice": final_result["messages"][-1].content}

        elif option == "Pulmonologist":
            async for chunk in pulmonologist_graph.astream(inputs, stream_mode="values"):
                final_result = chunk
            return {"pulmonologist_prediction": final_result["messages"][-1].content}

        elif option == "Psychologist":
            async for chunk in psychologist_graph.astream(inputs, stream_mode="values"):
                final_result = chunk
            return {"psychological_report": final_result["messages"][-1].content}

        elif option == "MultidisciplinaryTeam":
            try:
                set_ret = RunnableParallel({"context": retriever, "query": RunnablePassthrough()})
                rag_chain = set_ret | multidisciplinary_prompt | chat | StrOutputParser()
                report = await rag_chain.arun(query)
                return {"report": report}
            except Exception as e:
                print(f"Error during report generation: {e}")
                raise HTTPException(status_code=500, detail="Internal server error during report generation.")

        else:
            raise HTTPException(status_code=400, detail="Invalid option provided.")
    except Exception as e:
        print(f"Error in healthcare_assist endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error in healthcare_assist endpoint.")
