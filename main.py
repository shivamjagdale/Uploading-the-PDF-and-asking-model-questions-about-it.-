# Here you can execute this in your coding environment

import PyPDF2
from langchain_community.llms import Ollama  # Ensure you have the Ollama client installed


def read_pdf(file_path):
    """Read text from a PDF file."""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text


def ask_model(llm, question, context):
    """Ask a question to the llama3:instruct model."""
    # Wrap the question and context in a list
    prompts = [f"Context: {context}\nQuestion: {question}"]

    # Use the generate method with the prompts list
    response = llm.generate(prompts)

    # Extract the text from the response
    if response.generations and len(response.generations) > 0:
        return response.generations[0][0].text  # Access the text attribute
    else:
        return "No answer found."


def main():
    """Main function to run the PDF reader and question-answering."""
    pdf_path = 'C:/Users/Shivam/Downloads/NIPS-2017-attention-is-all-you-need-Paper.pdf'  # Specify your PDF file path
    text = read_pdf(pdf_path)

    # Initialize the model
    llm = Ollama(model="llama3:instruct", temperature=0)

    while True:
        question = input("Ask a question based on the PDF text (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_model(llm, question, text)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
