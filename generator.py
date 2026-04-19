from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


LLM_NAME = "google/flan-t5-base"


class AnswerGenerator:
    def __init__(self, model_name: str = LLM_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_qa_prompt(self, question: str, contexts):
        joined_context = "\n\n".join(
            [f"Context {i + 1}: {ctx}" for i, ctx in enumerate(contexts)]
        )

        prompt = f"""
You are an academic document question-answering assistant.

Answer only using the provided context.

Strict rules:
1. Do not use outside knowledge.
2. If the answer is not clearly supported by the context, say exactly:
   "Answer not found in the provided document."
3. Combine relevant information from multiple contexts when needed.
4. For definition/explanation questions, structure the answer as:
   - Definition
   - Explanation
   - Key details / prevention / implications
5. Use complete sentences.
6. Be precise and concise.
7. Quote short exact phrases from the context when useful, but do not copy long passages.

Context:
{joined_context}

Question:
{question}

Answer:
"""
        return prompt.strip()

    def build_summary_prompt(self, text: str):
        prompt = f"""
Summarize the following academic content clearly and concisely in 5-7 sentences.
Focus on the core concepts, major ideas, and important technical points.
Do not add outside information.

Text:
{text}

Summary:
"""
        return prompt.strip()

    def generate_text(self, prompt: str, max_new_tokens: int = 180):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def answer_question(self, question: str, contexts):
        prompt = self.build_qa_prompt(question, contexts)
        return self.generate_text(prompt, max_new_tokens=180)

    def summarize_text(self, text: str):
        prompt = self.build_summary_prompt(text)
        return self.generate_text(prompt, max_new_tokens=200)