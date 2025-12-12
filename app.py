from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *
import os

## init the memory intgreator of memory buffer
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=True
)


app = Flask(__name__)

dotenv_path = r"C:\Linkdin projects\Medical - system\Medical-system\.env"  # Use raw string for Windows paths
load_dotenv(dotenv_path=dotenv_path)

index_name = "medical-chatbot-abdelrhman"  

embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

chatModel = ChatOpenAI(model="gpt-4o-mini")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Here is the previous conversation context:\n{history}\n\nNow answer the user's new question:\n{input}")
    ]
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs={"k" : 3}) # top 3 k

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST']) # get method for taking from messege box
def chat():
    msg = request.form['msg'] # getting from a div called msg form the html
    input = msg
    print(input)

    #### memory intgraton #######################
    #memorizing the user query
    memory.chat_memory.add_user_message(msg)
    #############################################
    response = rag_chain.invoke({
        "input" : msg,
        "history" : memory.chat_memory.messages })
    print("Response: ",response['answer'])

    #### memory intgraton #######################
    #memorizing the Ai answer
    memory.chat_memory.add_ai_message(response['answer'])
    #############################################

    return str(response['answer'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)