import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()

# Works locally (.env) and on Streamlit Cloud (Secrets)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 Document Chatbot")

if not api_key:
    st.error("GROQ_API_KEY not found. Add it to .env (local) or Streamlit Secrets (cloud).")
    st.stop()


@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore", embeddings, allow_dangerous_deserialization=True
    )


@st.cache_resource
def load_llm():
    # If this model name is retired, pick a current one from the Groq console.
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key, temperature=0)


db = load_db()
llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask something about the document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks
    results = db.similarity_search(question, k=3)
    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt).content
        st.markdown(answer)
        with st.expander("Retrieved context"):
            st.write(context)

    st.session_state.messages.append({"role": "assistant", "content": answer})
