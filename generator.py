from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


LLM_NAME = "google/flan-t5-base"


class AnswerGenerator:
    def __init__(self, model_name: str = LLM_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_qa_prompt(self, question: str, contexts):
        """
        Build a grounded QA prompt using retrieved chunks.
        """
        joined_context = "\n\n".join(
            [f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)]
        )

        prompt = f"""
Answer the question using only the provided context.
If the answer is not present in the context, say: "Answer not found in the provided document."

Context:
{joined_context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    def build_summary_prompt(self, text: str):
        """
        Build summarization prompt.
        """
        prompt = f"""
Summarize the following academic content clearly and concisely in 5-7 sentences.

Text:
{text}

Summary:
"""
        return prompt.strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 180):
        """
        Generate output using FLAN-T5.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, question: str, contexts):
        prompt = self.build_qa_prompt(question, contexts)
        return self.generate_text(prompt, max_new_tokens=180)

    def summarize_text(self, text: str):
        prompt = self.build_summary_prompt(text)
        return self.generate_text(prompt, max_new_tokens=200)