from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from pdf_utils import extract_text_from_pdf
from embedding_utils import TextIndex
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_index = TextIndex()

# Pydantic model for request body
class QuestionRequest(BaseModel):
    question: str

@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "PDF Q&A API is running"}

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_location = os.path.join(upload_dir, file.filename)

        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text and create index
        text_pages = extract_text_from_pdf(file_location)
        
        if not text_pages or all(not text.strip() for text in text_pages.values()):
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        current_index.create_index(text_pages)

        return {
            "message": "PDF uploaded and indexed successfully.",
            "filename": file.filename,
            "pages": len(text_pages)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    try:
        if current_index.index is None:
            raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a PDF first.")

        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        answer = current_index.query(request.question)
        
        return {
            "question": request.question,
            "answer": answer,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Optional: Add endpoint to get document info
@app.get("/document-info/")
async def get_document_info():
    if current_index.index is None:
        return {"uploaded": False}
    
    return {
        "uploaded": True,
        "chunks": len(current_index.text_chunks) if current_index.text_chunks else 0
    }