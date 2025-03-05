import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import mimetypes
from docx import Document
import pandas as pd

temp_file_path = "temp_uploaded_file"
load_dotenv()

# Initialize Google Gemini client
def init_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set the GEMINI_API_KEY environment variable.")
        st.stop()
    return genai.Client(api_key=api_key)

# Extract text from .docx file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Extract data from .xls or .xlsx file
def extract_text_from_xls(file_path):
    df = pd.read_excel(file_path, sheet_name=None)
    text = ""
    for sheet, data in df.items():
        text += f"\nSheet: {sheet}\n"
        text += data.to_string(index=False)
    return text

# Extract data from .csv file
def extract_text_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string(index=False)

# Generate content using Gemini
def generate_content(client, input_text, user_query):
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_query),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )

    output_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        output_text += chunk.text

    return output_text

# Streamlit App UI
def main():
    st.title("Gemini Document Extractor")

    # File upload
    uploaded_file = st.file_uploader("Upload a document (PDF, Word, Excel, CSV, Text)",
                                     type=["pdf", "docx", "xls", "xlsx", "csv", "txt"])
    user_query = st.text_area("Enter what information to extract",
                              placeholder="E.g., Extract education, experience, skills...")

    if st.button("Extract Information"):
        if not uploaded_file:
            st.error("Please upload a document first.")
            return

        if not user_query.strip():
            st.error("Please enter the query you want to extract.")
            return

        # Determine file type and save temporarily
        file_extension = uploaded_file.name.split(".")[-1]
        temp_filename = f"{temp_file_path}.{file_extension}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())

        client = init_gemini_client()
        extracted_text = ""

        if file_extension == "docx":
            extracted_text = extract_text_from_docx(temp_filename)
        elif file_extension in ["xls", "xlsx"]:
            extracted_text = extract_text_from_xls(temp_filename)
        elif file_extension == "csv":
            extracted_text = extract_text_from_csv(temp_filename)
        else:
            with open(temp_filename, "r", encoding="utf-8") as f:
                extracted_text = f.read()

        # Generate content
        result = generate_content(client, extracted_text, user_query)

        # Display result
        st.subheader("Extracted Information:")
        st.markdown(result)

        # Cleanup
        os.remove(temp_filename)

if __name__ == "__main__":
    main()
