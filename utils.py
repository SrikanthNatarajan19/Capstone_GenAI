from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract text from a PDF uploaded through Streamlit.
    """
    try:
        reader = PdfReader(uploaded_file)
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        return "\n".join(pages_text)

    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - remove extra spaces
    - remove extra newlines
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    cleaned = " ".join(lines)
    cleaned = " ".join(cleaned.split())
    return cleaned


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split text into overlapping chunks.
    Character-based chunking keeps it simple and explainable.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)

        if end == text_len:
            break

        start += chunk_size - overlap

    return chunks