import pdfplumber
from typing import Dict

def extract_text_by_page(file) -> Dict[int, str]:
    """
    Extracts text from each page of a PDF and returns a dictionary: {page_number: text}
    """
    text_by_page = {}
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text_by_page[i + 1] = text.strip()
    return text_by_page
