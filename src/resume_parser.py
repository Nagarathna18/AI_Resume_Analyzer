import pymupdf
from docx import Document


def extract_text_from_pdf(file_path):
    """Extract text from a PDF resume."""
    
    text = ""

    document = pymupdf.open(file_path)

    for page in document:
        text += page.get_text()

    document.close()

    return text


def extract_text_from_docx(file_path):
    """Extract text from a DOCX resume."""
    
    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
    )

    return text


def extract_resume_text(file_path):
    """Extract resume text based on file type."""
    
    file_path = str(file_path).lower()

    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)

    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use PDF, DOCX, or TXT."
        )