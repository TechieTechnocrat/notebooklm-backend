import PyPDF2
import fitz  # PyMuPDF - better for text extraction
import io
import re

def extract_text_from_pdf(file_path):
    """Extract text from PDF with fallback methods"""
    text_pages = {}
    
    try:
        # Method 1: Try PyMuPDF first (better text extraction)
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                text_pages[f"page_{page_num + 1}"] = text
        doc.close()
        
        if text_pages:
            return text_pages
    
    except Exception as e:
        print(f"PyMuPDF failed: {e}, trying PyPDF2...")
    
    try:
        # Method 2: Fallback to PyPDF2
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_pages[f"page_{page_num + 1}"] = text
    
    except Exception as e:
        print(f"PyPDF2 also failed: {e}")
        raise Exception("Could not extract text from PDF using any method")
    
    if not text_pages:
        raise Exception("No text content found in the PDF")
    
    return text_pages

def clean_text(text):
    """Clean extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\.,!?;:()\-]', '', text)
    return text.strip()