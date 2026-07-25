import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama


def create_qa_chain(vectorstore):
    llm = ChatOllama(model="llama3", temperature=0.3,
                     base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=False
    )
