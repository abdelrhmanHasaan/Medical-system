import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Extract Data From the PDF File
def load_pdf_file(data: str):
    logger.info(f"Loading PDF files from directory: {data}")
    try:
        loader = DirectoryLoader(data,
                                 glob="*.pdf",
                                 loader_cls=PyPDFLoader)

        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files from {data}: {e}")
        raise

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    logger.info(f"Filtering {len(docs)} documents to minimal docs.")
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Split the Data into Text Chunks
def text_split(extracted_data: List[Document]):
    logger.info("Splitting text into chunks.")
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)
        logger.info(f"Successfully split into {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        raise

# Download the Embeddings from HuggingFace
def download_hugging_face_embeddings():
    logger.info("Downloading HuggingFace embeddings model.")
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  # this model return 384 dimensions
        logger.info("Successfully downloaded embeddings model.")
        return embeddings
    except Exception as e:
        logger.error(f"Error downloading embeddings model: {e}")
        raise
