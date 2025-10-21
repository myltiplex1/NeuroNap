import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuronap.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")
if not os.environ["GOOGLE_API_KEY"]:
    logger.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY not found in .env file")
logger.info("Loaded environment variables")

# Initialize embeddings and text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Initialized sentence-transformers embeddings")
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    raise

def create_knowledge_embeddings():
    logger.info("Starting creation of knowledge embeddings")
    knowledge_files = glob.glob("knowledge/*.pdf")
    if not knowledge_files:
        logger.error("No .pdf files found in 'knowledge' folder")
        raise FileNotFoundError("No .pdf files found in 'knowledge' folder.")
    
    all_docs = []
    for file_path in knowledge_files:
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)
            all_docs.extend(texts)
            logger.info(f"Successfully loaded and split {file_path} with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not all_docs:
        logger.error("No valid documents loaded from knowledge folder")
        raise ValueError("No valid documents loaded from knowledge folder.")
    
    logger.info(f"Creating FAISS index with {len(all_docs)} document chunks")
    try:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local("faiss_index")
        logger.info("FAISS embeddings created and saved to 'faiss_index'")
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise

def load_vectorstore():
    logger.info("Loading FAISS vectorstore")
    if not os.path.exists("faiss_index"):
        logger.error("FAISS index not found")
        raise FileNotFoundError("FAISS index not found. Run create_knowledge_embeddings first.")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    logger.info("FAISS vectorstore loaded successfully")
    return vectorstore

def update_vectorstore_with_user_results(user_results):
    logger.info("Updating vectorstore with user results")
    vectorstore = load_vectorstore()
    with open("user_results.txt", "w", encoding="utf-8") as f:
        f.write(user_results)
    user_loader = TextLoader("user_results.txt")
    user_docs = user_loader.load()
    user_texts = text_splitter.split_documents(user_docs)
    vectorstore.add_documents(user_texts)
    vectorstore.save_local("faiss_index")
    logger.info("User results added to vectorstore and saved")
    return vectorstore

def setup_rag():
    logger.info("Setting up RAG chain")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
        logger.info("Initialized Gemini 2.0 Flash LLM")
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}. Check GEMINI_API_KEY or quota")
        raise
    vectorstore = load_vectorstore()
    prompt_template = """\
You are an expert assistant in EEG-based sleep analysis. Use the provided context from sleep stage definitions, EEG characteristics, and user-specific results to answer the query.
Context: {context}
User Query: {question}
Provide a clear, educational response in plain text, avoiding Markdown symbols (e.g., **, *). Use bullet points (•) for lists and exact values from user data if referenced. Include concise explanations for each metric or finding.
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    logger.info("RAG chain setup completed")
    return qa_chain

def chat_with_llm(query, qa_chain):
    logger.info(f"Processing query: {query}")
    try:
        result = qa_chain({"query": query})["result"]
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {e}. Check GEMINI_API_KEY or quota"

if __name__ == "__main__":
    logger.info("Running rag.py as main script")
    os.makedirs("knowledge", exist_ok=True)
    create_knowledge_embeddings()