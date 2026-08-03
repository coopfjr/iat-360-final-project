from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the SFU Data
print("Loading data...")
loader = DirectoryLoader('./sfu_data', glob="**/*.txt")
documents = loader.load()

# 2. Split the text
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 3. Create the Database
print("Creating database...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Connect to your local Llama 3.1
print("Connecting to Llama...")
llm = ChatOllama(model="llama3.1")

# 5. Create the modern QA Chain (Using classic package)
system_prompt = (
    "You are an academic advisor for SFU SIAT. "
    "Use the following pieces of retrieved context to answer the student's question. "
    "If you don't know the answer based on the context, say that you don't know. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 3}), question_answer_chain)

# 6. Run the Evaluation Test!
print("\n--- Model Response ---")
query = "I want to take IAT 265. I have completed IAT 167 with a C-, but I failed MATH 130. Can I take it?"
response = qa_chain.invoke({"input": query})
print(response['answer'])