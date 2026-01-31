from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load PDF
loader = PyPDFLoader("/workspaces/Project-IA-PDF-Doc-Assistant/inputs/diapo.pdf")
docs = loader.load()

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3. Vector DB
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = Chroma.from_documents(chunks, embeddings)

# 4. LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 5. QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever()
)

# 6. Ask
pregunta = "¿Cuál es la metodología principal descrita en este PDF?"
respuesta = qa_chain.run(pregunta)

print(respuesta)
