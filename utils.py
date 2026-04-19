import re
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
    Basic text cleaning for academic documents.
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def split_into_sentences(text: str):
    """
    Simple sentence splitter suitable for academic prose.
    """
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40):
    """
    Split text into overlapping chunks using sentence-aware grouping.
    Smaller chunks improve retrieval precision for short academic passages.
    """
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if current_len + sent_len + 1 <= chunk_size:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
            current_len = len(current_chunk)
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            if overlap > 0 and chunks:
                prev_chunk = chunks[-1]
                # Split by words to avoid cutting characters in half
                prev_words = prev_chunk.split()
                overlap_text = " ".join(prev_words[-overlap:]) if len(prev_words) > overlap else prev_chunk
                current_chunk = (overlap_text.strip() + " " + sentence).strip()
            else:
                current_chunk = sentence.strip()

            current_len = len(current_chunk)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks