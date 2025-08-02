from fastapi import FastAPI, UploadFile, File
import shutil
import os
from pdf_utils import extract_text_from_pdf
from embedding_utils import TextIndex

app = FastAPI()
current_index = TextIndex()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text_pages = extract_text_from_pdf(file_location)
    current_index.create_index(text_pages)

    return {"message": "PDF uploaded and indexed successfully."}

@app.post("/ask/")
async def ask_question(question: str):
    if current_index.index is None:
        return {"error": "No document uploaded yet."}

    answer = current_index.query(question)
    return {"answer": answer}
