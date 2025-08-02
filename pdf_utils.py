import PyPDF2

def extract_text_from_pdf(file_path):
    pdf = PyPDF2.PdfReader(file_path)
    text_pages = {}
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        text_pages[i + 1] = text if text else ""
    return text_pages
