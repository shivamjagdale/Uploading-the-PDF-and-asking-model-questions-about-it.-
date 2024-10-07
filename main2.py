# This is for streamlit app.py

import streamlit as st
import PyPDF2
from langchain_community.llms import Ollama  # Ensure you have the Ollama client installed


def read_pdf(file):
    """Read text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text


def ask_model(llm, question, context):
    """Ask a question to the llama3:instruct model."""
    prompts = [f"Context: {context}\nQuestion: {question}"]
    response = llm.generate(prompts)

    if response.generations and len(response.generations) > 0:
        return response.generations[0][0].text
    else:
        return "No answer found."


def main():
    """Main function to run the Streamlit app."""
    st.title("PDF Question Answering")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read the PDF file
        text = read_pdf(uploaded_file)
        st.write("PDF content extracted successfully.")

        # Initialize the model
        llm = Ollama(model="llama3:instruct", temperature=0)

        # User input for question
        question = st.text_input("Ask a question based on the PDF content:")

        if st.button("Get Answer"):
            if question:
                answer = ask_model(llm, question, text)
                st.write("Answer:", answer)
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
