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


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    """
    Split text into overlapping chunks, trying to end on sentence boundaries
    when possible.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            period_pos = text.rfind(".", start, end)
            newline_pos = text.rfind("\n", start, end)
            split_pos = max(period_pos, newline_pos)
            if split_pos > start + 100:
                end = split_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_len:
            break

        start = max(end - overlap, start + 1)

    return chunks