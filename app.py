import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

st.set_page_config(page_title="PDF Q&A using RAG", layout="wide")

st.title("📄 PDF Question Answering System (RAG)")
st.write("Ask questions strictly based on the uploaded PDF.")

# --------------------------------------------------
# Initialize LLM (ONCE)
# --------------------------------------------------
llm = ChatOpenAI(temperature=0)

# --------------------------------------------------
# Upload PDF
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Load PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=80
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings & vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )

        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        st.success("PDF processed successfully!")

    # --------------------------------------------------
    # Prompt Template
    # --------------------------------------------------
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer the question using ONLY the context below.
If the answer is not present in the context, respond exactly with:
"out of syllabus"

Context:
{context}

Question:
{question}
"""
    )

    chain = prompt_template | llm

    # --------------------------------------------------
    # Question Input
    # --------------------------------------------------
    user_query = st.text_input("Enter your question")

    if user_query:
        docs = retriever.get_relevant_documents(user_query)

        if not docs:
            st.error("out of syllabus")
        else:
            context = "\n\n".join(doc.page_content for doc in docs)

            response = chain.invoke({
                "context": context,
                "question": user_query
            })

            st.subheader("Answer")
            st.write(response.content)

            # Optional: show sources
            with st.expander("📌 Show Source Pages"):
                for i, doc in enumerate(docs):
                    st.write(f"Source {i+1} — Page {doc.metadata.get('page')}")

else:
    st.info("Please upload a PDF to start.")
