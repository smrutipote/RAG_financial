from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient

from PyPDF2 import PdfReader
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Load and chunk all PDFs
def load_and_split_pdfs(pdf_folder="data"):
    pdf_folder = Path(pdf_folder)
    all_texts = []

    for pdf_file in pdf_folder.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        all_texts.append(text)

    combined_text = "\n\n".join(all_texts)

    # Wrap text in LangChain Document and split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    doc = Document(page_content=combined_text, metadata={"source": "combined_pdfs"})
    chunks = splitter.split_documents([doc])
    return chunks

# Create or load FAISS vector index
def get_vectorstore(chunks, persist_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(persist_path):
        return FAISS.load_local(
            persist_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(persist_path)
        return vs

# Manually call Mistral model via InferenceClient
def call_mistral(prompt):
    client = InferenceClient(
        model="HuggingFaceH4/zephyr-7b-beta",
        token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    response = client.text_generation(
        prompt=f"<s>[INST] {prompt.strip()} [/INST]",
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )

    return response.strip()

# Retrieve relevant chunks and ask Mistral
def query_rag(query):
    chunks = load_and_split_pdfs()
    vectorstore = get_vectorstore(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build the prompt for InferenceClient
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = call_mistral(prompt)
    sources = [doc.metadata.get("source", "N/A") for doc in docs]

    return answer, sources
