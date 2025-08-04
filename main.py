from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdf_utils import extract_text_by_page
from grok_embedder import GrokEmbedder
from pydantic import BaseModel


app = FastAPI()


class QuestionRequest(BaseModel):
    question: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = GrokEmbedder()
current_pdf = {}  

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF file, extracts text, and builds an index for question answering.
    """
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text_by_page = extract_text_by_page("temp.pdf")

    if not text_by_page:
        return JSONResponse({"error": "Could not extract text from PDF"}, status_code=400)

    current_pdf["filename"] = file.filename
    current_pdf["text_by_page"] = text_by_page
    embedder.create_index(text_by_page)

    return {"message": f"{file.filename} uploaded and indexed successfully."}

@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    """
    Accepts a JSON payload with a question field and returns an answer with citation.
    """
    if not current_pdf.get("text_by_page"):
        return JSONResponse({"error": "No PDF uploaded yet."}, status_code=400)

    question = payload.question
    result = embedder.analyze_and_answer(question)
    return {"answer": result["answer"], "citation": result["citation"]}
@app.post("/reset")
async def reset():
    """
    Clears the in-memory state.
    """
    current_pdf.clear()
    embedder.text_chunks.clear()
    embedder.entities.clear()
    return {"message": "Session reset."}
