import os
import logging
from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from src.prompt import *

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables securely
load_dotenv()

# Check for required environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    logger.error("Missing required environment variables (PINECONE_API_KEY, OPENAI_API_KEY). Ensure .env is set up correctly.")

# Ensure os.environ is populated for LangChain's internal use
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""

app = Flask(__name__)

# Init the memory integrator of memory buffer
memory = ConversationBufferMemory(
    memory_key="history",
    input_key="input",
    return_messages=True
)

rag_chain = None

def init_rag_chain():
    global rag_chain
    try:
        logger.info("Initializing RAG chain components...")
        index_name = "medical-chatbot-abdelrhman"
        embeddings = download_hugging_face_embeddings()

        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        chatModel = ChatOpenAI(model="gpt-4o-mini")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Here is the previous conversation context:\n{history}\n\nNow answer the user's new question:\n{input}")
            ]
        )

        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # top 3 k

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        logger.info("RAG chain successfully initialized.")
    except Exception as e:
        logger.error(f"Error initializing RAG chain: {e}", exc_info=True)

# Initialize the chain when app starts
init_rag_chain()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def chat():
    # Input validation
    if 'msg' not in request.form:
        logger.warning("Received chat request without 'msg' field.")
        return jsonify({"error": "Bad Request: 'msg' field is required."}), 400

    msg = request.form['msg'].strip()
    if not msg:
        logger.warning("Received empty message.")
        return jsonify({"error": "Bad Request: 'msg' cannot be empty."}), 400

    if len(msg) > 1000:
        logger.warning("Received message exceeding length limit.")
        return jsonify({"error": "Bad Request: 'msg' is too long."}), 400

    logger.info(f"User Query: {msg}")

    if not rag_chain:
        logger.error("RAG chain is not initialized, cannot process request.")
        return "Sorry, the chatbot is currently unavailable due to an internal error.", 500

    try:
        # Memorize the user query
        memory.chat_memory.add_user_message(msg)

        # Invoke RAG chain
        response = rag_chain.invoke({
            "input": msg,
            "history": memory.chat_memory.messages
        })
        answer = response.get('answer', "I couldn't find an answer.")

        logger.info(f"AI Response: {answer}")

        # Memorize the AI answer
        memory.chat_memory.add_ai_message(answer)

        return str(answer)
    except Exception as e:
        logger.error(f"Error during RAG chain invocation: {e}", exc_info=True)
        return "Sorry, an error occurred while processing your request.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
