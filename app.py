import json
import time
import streamlit as st

from utils import extract_text_from_pdf, clean_text, chunk_text
from retriever import SemanticRetriever
from generator import AnswerGenerator
from evaluator import get_memory_usage_mb, token_f1_score, rouge_scores


st.set_page_config(page_title="Academic QA Assistant", layout="wide")
st.info("""
How to use this app:
1. Upload a PDF or paste text
2. Click 'Process Document'
3. Ask a question or generate a summary
4. Review retrieved chunks and evaluation metrics
""")


@st.cache_resource
def load_retriever():
    return SemanticRetriever()


@st.cache_resource
def load_generator():
    return AnswerGenerator()


def load_sample_questions():
    try:
        with open("sample_questions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def main():
    st.title("Academic Question Answering using RAG + FLAN-T5")
    st.write("Upload a PDF or paste academic text, then ask questions grounded in the document.")

    retriever = load_retriever()
    generator = load_generator()

    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "index_built" not in st.session_state:
        st.session_state.index_built = False

    st.sidebar.header("Settings")
    chunk_size = st.sidebar.slider("Chunk Size", 300, 1000, 500, 50)
    overlap = st.sidebar.slider("Chunk Overlap", 50, 300, 100, 10)
    top_k = st.sidebar.slider("Top-K Retrieval", 1, 5, 3, 1)

    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    pasted_text = st.text_area("Or paste academic text here", height=200)

    if st.button("Process Document"):
        raw_text = ""

        if uploaded_pdf is not None:
            try:
                raw_text += extract_text_from_pdf(uploaded_pdf)
            except Exception as e:
                st.error(str(e))
                return

        if pasted_text.strip():
            raw_text += "\n" + pasted_text.strip()

        cleaned = clean_text(raw_text)

        if not cleaned:
            st.warning("No readable text found. Please upload a valid PDF or paste text.")
            return

        chunks = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)

        if not chunks:
            st.warning("Chunking failed. No chunks were created.")
            return

        start_index_time = time.perf_counter()
        retriever.build_index(chunks)
        index_time = time.perf_counter() - start_index_time

        st.session_state.document_text = cleaned
        st.session_state.chunks = chunks
        st.session_state.index_built = True

        st.success("Document processed successfully.")
        st.write(f"**Characters extracted:** {len(cleaned)}")
        st.write(f"**Number of chunks:** {len(chunks)}")
        st.write(f"**Index build time:** {index_time:.3f} seconds")
        st.write(f"**Memory usage:** {get_memory_usage_mb():.2f} MB")

    if st.session_state.index_built:
        st.subheader("Ask a Question")
        question = st.text_input("Enter your question")

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
                return

            retrieval_start = time.perf_counter()
            results = retriever.retrieve(question, top_k=top_k)
            if not results or results[0]['score'] < 0.20:
                st.subheader("Generated Answer")
                st.write("Answer not found in the provided document.")
                return
            retrieval_time = time.perf_counter() - retrieval_start

            contexts = [r["text"] for r in results]

            generation_start = time.perf_counter()
            answer = generator.answer_question(question, contexts)
            generation_time = time.perf_counter() - generation_start

            total_time = retrieval_time + generation_time

            st.subheader("Generated Answer")
            st.write(answer)

            st.subheader("Performance Metrics")
            st.write(f"**Retrieval time:** {retrieval_time:.3f} seconds")
            st.write(f"**Generation time:** {generation_time:.3f} seconds")
            st.write(f"**Total time:** {total_time:.3f} seconds")
            st.write(f"**Memory usage:** {get_memory_usage_mb():.2f} MB")

            with st.expander("Retrieved Chunks"):
                for i, result in enumerate(results, start=1):
                    st.markdown(f"**Chunk {i}** | Score: {result['score']:.4f}")
                    st.write(result["text"])
                    st.markdown("---")

        st.subheader("Generate Summary")
        if st.button("Generate Document Summary"):
            doc_preview = st.session_state.document_text[:2500]

            summary_start = time.perf_counter()
            summary = generator.summarize_text(doc_preview)
            summary_time = time.perf_counter() - summary_start

            st.write(summary)
            st.write(f"**Summary generation time:** {summary_time:.3f} seconds")

        st.subheader("Mini Evaluation")
        sample_qas = load_sample_questions()

        if st.button("Run Sample Evaluation"):
            if not sample_qas:
                st.warning("No sample questions found.")
                return

            all_scores = []

            for item in sample_qas:
                q = item["question"]
                expected = item["expected_answer"]

                results = retriever.retrieve(q, top_k=top_k)
                contexts = [r["text"] for r in results]
                generated = generator.answer_question(q, contexts)

                f1 = token_f1_score(generated, expected)
                all_scores.append(f1)

                with st.expander(f"Question: {q}"):
                    st.write(f"**Expected:** {expected}")
                    st.write(f"**Generated:** {generated}")
                    st.write(f"**F1 Score:** {f1:.4f}")

            avg_f1 = sum(all_scores) / len(all_scores) if all_scores else 0.0
            st.write(f"**Average F1 Score:** {avg_f1:.4f}")

        with st.expander("View Extracted Document Text"):
            st.write(st.session_state.document_text[:5000])


if __name__ == "__main__":
    main()