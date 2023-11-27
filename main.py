from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def main():
    print("\n\nWelcome to LangChain Project!\n\n")
    load_dotenv()
    st.set_page_config(page_title="Ask the PDF!")
    st.header("Ask the PDF!")

    # Upload the file.
    pdf = st.file_uploader("Upload your PDF here: ", type="pdf")

    # Extract the pdf's text.
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("\n Ask a Question about your PDF!")
        if user_question:
            query_embedding = embeddings.embed_query(user_question)
            docs = knowledge_base.similarity_search(query_embedding, top_k=5)







if __name__ == "__main__":
    main()


