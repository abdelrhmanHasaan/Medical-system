from dotenv import load_dotenv
import os
import logging
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting index storage process...")
    # Load environment variables
    load_dotenv()

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is missing in environment variables.")
        return
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is missing in environment variables.")
        return

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Update path to a relative path
    data_path = './data'
    if not os.path.exists(data_path):
        logger.error(f"Data directory '{data_path}' not found.")
        return

    try:
        extracted_data = load_pdf_file(data=data_path)
        filter_data = filter_to_minimal_docs(extracted_data)
        text_chunks = text_split(filter_data)
        embeddings = download_hugging_face_embeddings()
    except Exception as e:
        logger.error(f"Error during data processing or embedding downloading: {e}")
        return

    try:
        logger.info("Initializing Pinecone connection...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "medical-chatbot-abdelrhman"

        if not pc.has_index(index_name):
            logger.info(f"Index '{index_name}' does not exist. Creating...")
            pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists.")

        index = pc.Index(index_name)

        logger.info("Storing documents in Pinecone VectorStore...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        logger.info("Documents stored successfully.")
    except Exception as e:
        logger.error(f"Error communicating with Pinecone or storing documents: {e}")

if __name__ == "__main__":
    main()
