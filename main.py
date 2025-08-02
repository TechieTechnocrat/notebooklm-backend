from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import shutil
import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import traceback

# Import your enhanced modules
from pdf_utils import extract_text_from_pdf, validate_resume_content
from embedding_utils import EnhancedTextIndex

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced PDF Resume Q&A API",
    description="Advanced PDF document analysis with intelligent Q&A capabilities",
    version="2.0.0"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global variables
current_index = EnhancedTextIndex()
current_document_info = {}

# Enhanced Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question about the document")
    include_context: bool = Field(default=False, description="Include source context in response")

class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    document_type: str
    entities_found: Dict[str, int]
    processing_time: float

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: Optional[str] = None
    sources: Optional[List[str]] = None
    processing_time: float
    status: str

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status": "error"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status": "error"}
    )

@app.get("/")
async def health_check():
    """Enhanced health check with system status"""
    return {
        "status": "healthy",
        "message": "Enhanced PDF Q&A API is running",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "document_loaded": current_index.index is not None,
        "available_endpoints": [
            "/upload-pdf/",
            "/ask/",
            "/ask-specific/",
            "/document-info/",
            "/document-summary/"
        ]
    }

@app.post("/upload-pdf/", response_model=UploadResponse)
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Enhanced PDF upload with better validation and processing"""
    start_time = datetime.now()
    
    try:
        # Enhanced file validation
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        if file.size and file.size > 50 * 1024 * 1024:  # 50MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")
        
        # Create uploads directory
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_location = os.path.join(upload_dir, safe_filename)

        # Save uploaded file
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        logger.info(f"File saved: {file_location}")

        # Extract text with enhanced method
        try:
            text_pages = extract_text_from_pdf(file_location)
        except Exception as e:
            # Clean up file on extraction failure
            if os.path.exists(file_location):
                os.remove(file_location)
            raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")
        
        # Validate content
        if not text_pages or all(not text.strip() for text in text_pages.values()):
            if os.path.exists(file_location):
                os.remove(file_location)
            raise HTTPException(status_code=400, detail="No readable text found in the PDF")
        
        # Validate if it looks like a resume
        is_resume = validate_resume_content(text_pages)
        document_type = "resume" if is_resume else "document"
        
        # Create enhanced index
        try:
            current_index.create_index(text_pages)
            logger.info("Index created successfully")
        except Exception as e:
            if os.path.exists(file_location):
                os.remove(file_location)
            raise HTTPException(status_code=500, detail=f"Failed to create document index: {str(e)}")
        
        # Get document statistics
        stats = current_index.get_stats()
        
        # Store document info globally
        global current_document_info
        current_document_info = {
            "filename": file.filename,
            "upload_time": start_time.isoformat(),
            "pages": len(text_pages),
            "document_type": document_type,
            "stats": stats
        }
        
        # Schedule cleanup of old files in background
        background_tasks.add_task(cleanup_old_files, upload_dir)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return UploadResponse(
            message=f"PDF uploaded and indexed successfully as {document_type}",
            filename=file.filename,
            pages=len(text_pages),
            document_type=document_type,
            entities_found=stats.get("extracted_entities", {}),
            processing_time=processing_time
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/ask/", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Enhanced question answering with better context and confidence"""
    start_time = datetime.now()
    
    try:
        # Validate inputs
        if current_index.index is None:
            raise HTTPException(
                status_code=400, 
                detail="No document uploaded yet. Please upload a PDF first."
            )

        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Log the question for monitoring
        logger.info(f"Question received: {request.question}")

        # Get answer using enhanced method
        answer = current_index.query(request.question)
        
        # Determine confidence level based on answer characteristics
        confidence = "high"
        if "couldn't find" in answer.lower() or "no clear" in answer.lower():
            confidence = "low"
        elif "here's the relevant information" in answer.lower():
            confidence = "medium"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = QuestionResponse(
            question=request.question,
            answer=answer,
            confidence=confidence,
            processing_time=processing_time,
            status="success"
        )
        
        # Add context if requested
        if request.include_context:
            response.sources = [f"Document: {current_document_info.get('filename', 'Unknown')}"]
        
        logger.info(f"Question answered successfully in {processing_time:.2f}s")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/ask-specific/")
async def ask_specific_question(request: QuestionRequest):
    """Handle specific resume questions (name, education, etc.) with high accuracy"""
    start_time = datetime.now()
    
    try:
        if current_index.index is None:
            raise HTTPException(
                status_code=400,
                detail="No document uploaded yet. Please upload a PDF first."
            )

        logger.info(f"Specific question received: {request.question}")

        # Use the specific question handler
        answer = current_index.answer_specific_question(request.question)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "question": request.question,
            "answer": answer,
            "type": "specific_answer",
            "processing_time": processing_time,
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_specific_question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing specific question: {str(e)}")

@app.get("/document-info/")
async def get_document_info():
    """Get comprehensive document information"""
    try:
        if current_index.index is None:
            return {"uploaded": False, "message": "No document uploaded"}
        
        stats = current_index.get_stats()
        
        return {
            "uploaded": True,
            "document_info": current_document_info,
            "index_stats": stats,
            "capabilities": [
                "Answer questions about resume content",
                "Extract name, education, experience",
                "Identify skills and contact information",
                "Provide section-specific information"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return {"uploaded": False, "error": str(e)}

@app.get("/document-summary/")
async def get_document_summary():
    """Get a structured summary of the uploaded resume"""
    try:
        if current_index.index is None:
            raise HTTPException(status_code=400, detail="No document uploaded")
        
        summary = current_index.get_resume_summary()
        return {
            "summary": summary,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating summary: {str(e)}")

@app.delete("/clear-document/")
async def clear_document():
    """Clear the current document and free memory"""
    try:
        global current_index, current_document_info
        
        # Reset the index
        current_index = EnhancedTextIndex()
        current_document_info = {}
        
        logger.info("Document cleared successfully")
        return {"message": "Document cleared successfully", "status": "success"}
    
    except Exception as e:
        logger.error(f"Error clearing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing document: {str(e)}")

# Background tasks
async def cleanup_old_files(upload_dir: str, max_age_hours: int = 24):
    """Clean up old uploaded files"""
    try:
        current_time = datetime.now()
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")

# Additional utility endpoints
@app.get("/supported-questions/")
async def get_supported_questions():
    """Get examples of questions that work well with this system"""
    return {
        "general_questions": [
            "What is this person's name?",
            "What is their educational background?",
            "What work experience do they have?",
            "What are their technical skills?",
            "What is their contact information?",
            "Tell me about their projects",
            "What achievements are mentioned?",
            "What certifications do they have?"
        ],
        "specific_questions": [
            "What degree does this person have?",
            "Which companies have they worked for?",
            "How many years of experience do they have?",
            "What programming languages do they know?",
            "What is their email address?",
            "What university did they attend?"
        ],
        "tips": [
            "Be specific in your questions",
            "Ask about one topic at a time",
            "Use clear, simple language",
            "Questions about resume sections work best"
        ]
    }

@app.get("/health/")
async def detailed_health_check():
    """Detailed health check for monitoring"""
    try:
        # Check if models are loaded
        models_status = {
            "embedding_model": hasattr(current_index, 'embedder') and current_index.embedder is not None,
            "qa_model": hasattr(current_index, 'qa_model') and current_index.qa_model is not None,
            "index_ready": current_index.index is not None
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": models_status,
            "document_loaded": current_index.index is not None,
            "memory_usage": "OK",  # You could add actual memory monitoring here
            "uptime": "Available"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for different environments
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Set to False in production
        "log_level": "info"
    }
    
    logger.info("Starting Enhanced PDF Q&A API...")
    logger.info(f"Server will be available at http://{config['host']}:{config['port']}")
    logger.info("API Documentation available at http://localhost:8000/docs")
    
    uvicorn.run("main:app", **config)