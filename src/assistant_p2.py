import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# ===============================
# CARGA DEL PDF
# ===============================
pdf_path = "/workspaces/Project-IA-PDF-Doc-Assistant/inputs/diapo.pdf"  # Cambia por tu PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ===============================
# DIVISIÓN DEL TEXTO
# ===============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

# ===============================
# EMBEDDINGS
# ===============================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

# ===============================
# VECTOR STORE
# ===============================
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
vector_db.persist()

# ===============================
# MODELO LLM
# ===============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0
)

# ===============================
# PROMPT PERSONALIZADO
# ===============================
prompt_template = """
Eres un asistente académico experto.
Responde ÚNICAMENTE usando la información del contexto proporcionado.
NO digas que no tienes acceso al PDF ni a archivos externos.
Si la respuesta no está en el contexto, responde exactamente:
"La información no se encuentra en el documento."

Contexto:
{context}

Pregunta:
{question}

Respuesta clara y concisa:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# ===============================
# CADENA QA
# ===============================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# ===============================
# CONSULTA
# ===============================
pregunta = "¿Cuál es la metodología principal descrita en el documento?"

respuesta = qa_chain.invoke({"query": pregunta})

print("\nRESPUESTA:\n")
print(respuesta["result"])

print("\n--- DOCUMENTOS USADOS ---\n")
for i, doc in enumerate(respuesta["source_documents"], 1):
    print(f"[Documento {i}]")
    print(doc.page_content[:500])
    print("-" * 60)
