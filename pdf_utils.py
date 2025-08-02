import PyPDF2
import fitz  # PyMuPDF
import pdfplumber
import io
import re
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFExtractor:
    """Enhanced PDF text extraction with multiple fallback methods"""
    
    @staticmethod
    def extract_with_pymupdf(file_path: str) -> Dict[str, str]:
        """Extract text using PyMuPDF (best for most PDFs)"""
        text_pages = {}
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text with better formatting
                text_dict = page.get_text("dict")
                text_blocks = []
                
                for block in text_dict["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            if line_text.strip():
                                text_blocks.append(line_text.strip())
                
                page_text = "\n".join(text_blocks)
                if page_text.strip():
                    text_pages[f"page_{page_num + 1}"] = page_text
            
            doc.close()
            logger.info(f"PyMuPDF extracted {len(text_pages)} pages")
            return text_pages
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return {}

    @staticmethod
    def extract_with_pdfplumber(file_path: str) -> Dict[str, str]:
        """Extract text using pdfplumber (good for tables and complex layouts)"""
        text_pages = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with better formatting
                    text = page.extract_text()
                    
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        table_text = "\n"
                        for table in tables:
                            for row in table:
                                if row and any(cell for cell in row if cell):
                                    table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        text += table_text
                    
                    if text and text.strip():
                        text_pages[f"page_{page_num + 1}"] = text.strip()
            
            logger.info(f"pdfplumber extracted {len(text_pages)} pages")
            return text_pages
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return {}

    @staticmethod
    def extract_with_pypdf2(file_path: str) -> Dict[str, str]:
        """Extract text using PyPDF2 (fallback method)"""
        text_pages = {}
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        text_pages[f"page_{page_num + 1}"] = text.strip()
            
            logger.info(f"PyPDF2 extracted {len(text_pages)} pages")
            return text_pages
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            return {}

    @staticmethod
    def clean_and_normalize_text(text: str) -> str:
        """Enhanced text cleaning for better processing"""
        if not text:
            return ""
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Space between numbers and letters
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Space between letters and numbers
        
        # Fix bullet points and formatting
        text = re.sub(r'•', '* ', text)
        text = re.sub(r'[\u2022\u2023\u2043]', '* ', text)  # Various bullet characters
        
        # Fix excessive whitespace but preserve structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\t+', ' ', text)  # Tabs to spaces
        
        # Remove weird characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,!?;:()\-@#+*/\n\[\]{}|]', ' ', text)
        
        # Fix common resume formatting issues
        text = re.sub(r'(?i)(email|e-mail)\s*:?\s*', 'Email: ', text)
        text = re.sub(r'(?i)(phone|tel|mobile)\s*:?\s*', 'Phone: ', text)
        
        return text.strip()

def extract_text_from_pdf(file_path: str) -> Dict[str, str]:
    """
    Enhanced PDF text extraction with multiple methods and better error handling
    """
    extractor = EnhancedPDFExtractor()
    
    # Try extraction methods in order of preference
    extraction_methods = [
        ("pdfplumber", extractor.extract_with_pdfplumber),
        ("PyMuPDF", extractor.extract_with_pymupdf),
        ("PyPDF2", extractor.extract_with_pypdf2)
    ]
    
    best_result = {}
    best_method = None
    
    for method_name, method_func in extraction_methods:
        try:
            logger.info(f"Trying extraction with {method_name}")
            result = method_func(file_path)
            
            if result and len(result) > len(best_result):
                best_result = result
                best_method = method_name
                
                # If we got good results, we can break early
                total_chars = sum(len(text) for text in result.values())
                if total_chars > 500:  # Reasonable amount of text
                    logger.info(f"Good extraction achieved with {method_name}")
                    break
                    
        except Exception as e:
            logger.error(f"Method {method_name} failed: {e}")
            continue
    
    if not best_result:
        raise Exception("Could not extract text from PDF using any available method")
    
    # Clean and normalize all extracted text
    cleaned_result = {}
    for page_key, text in best_result.items():
        cleaned_text = extractor.clean_and_normalize_text(text)
        if cleaned_text and len(cleaned_text.strip()) > 20:  # Only keep substantial content
            cleaned_result[page_key] = cleaned_text
    
    if not cleaned_result:
        raise Exception("No substantial text content found in the PDF after cleaning")
    
    logger.info(f"Successfully extracted text using {best_method}: {len(cleaned_result)} pages")
    
    # Log sample of extracted text for debugging
    first_page = list(cleaned_result.values())[0]
    logger.info(f"Sample text (first 200 chars): {first_page[:200]}...")
    
    return cleaned_result

def validate_resume_content(text_dict: Dict[str, str]) -> bool:
    """Validate that the extracted text looks like a resume"""
    full_text = " ".join(text_dict.values()).lower()
    
    # Common resume indicators
    resume_keywords = [
        'experience', 'education', 'skills', 'work', 'employment',
        'university', 'college', 'degree', 'bachelor', 'master',
        'email', 'phone', 'contact', 'resume', 'cv'
    ]
    
    found_keywords = sum(1 for keyword in resume_keywords if keyword in full_text)
    
    # Check for email pattern
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', full_text))
    
    # Should have at least 3 resume keywords or an email
    is_valid = found_keywords >= 3 or has_email
    
    logger.info(f"Resume validation: {found_keywords} keywords found, email: {has_email}, valid: {is_valid}")