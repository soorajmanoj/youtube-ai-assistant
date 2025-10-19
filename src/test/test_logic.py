import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# --------------------
# 1️⃣ Load environment
# --------------------
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    print("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
    exit()

video_url = "https://www.youtube.com/watch?v=VunDCygsAsw"


# --------------------
# 2️⃣ Extract video ID
# --------------------
def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None


video_id = get_video_id(video_url)
if not video_id:
    print("Could not extract video ID from URL.")
    exit()

# --------------------
# 3️⃣ Fetch transcript
# --------------------
print(f"Fetching transcript for {video_id}...")
try:
    transcript_data = YouTubeTranscriptApi().fetch(video_id)
    transcript_text = " ".join([snippet.text for snippet in transcript_data])
    documents = [Document(page_content=transcript_text, metadata={"source": video_id})]
    print("Transcript fetched successfully.")
except Exception as e:
    print(f"Error fetching transcript: {e}")
    exit()

# --------------------
# 4️⃣ Create embeddings & FAISS vector store
# --------------------
print("Building vector index...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("Vector index built successfully.")

# --------------------
# 5️⃣ Create conversational QA chain
# --------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    verbose=False
)

# --------------------
# 6️⃣ Interactive Q&A loop
# --------------------
print("\n✅ YouTube Q&A Bot ready! Type your question below.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    question = input("You: ")
    if question.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    response = qa_chain.invoke({"question": question})
    print(f"Bot: {response['answer']}\n")
