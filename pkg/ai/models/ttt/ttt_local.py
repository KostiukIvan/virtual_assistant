import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline, pipeline

import pkg.config as config
from pkg.ai.models.ttt.memory_managers.rag.knowledge_base import KnowledgeBase
from pkg.ai.models.ttt.memory_managers.rag.rag_assisstant import RAGAssistant
from pkg.ai.models.ttt.memory_managers.sliding_window_memory_manager import MemoryManager
from pkg.ai.models.ttt.ttt_interface import TextToTextModel

logger = logging.getLogger(__name__)
hf_token = os.environ.get("HF_TOKEN")


class LocalTextToTextModel(TextToTextModel):
    def __init__(
        self,
        model: str = config.TTT_MODEL,
    ):
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)
        self.max_length = config.TTT_MAX_TOKENS
        self.num_return_sequences = config.TTT_NUM_RETURN_SEQUENCES
        self.memory_size = config.TTT_MEMORY_SIZE
        self.generator: Pipeline | None = None

        # memory
        self.memory = MemoryManager(
            window_size=config.TTT_MEMORY_MANAGER_WINDOW_SIZE, summarize_every=config.TTT_MEMORY_MANAGER_SUMMARY_EVERY
        )

        # knowledge base + RAG
        self.kb = KnowledgeBase()
        self.kb.add_documents(
            [
                "ACME Corp offers tax consultation, bookkeeping, and payroll services.",
                "Our office hours are Monday to Friday, 9am–6pm.",
                "You can book appointments online or by phone. Each consultation lasts 45 minutes.",
            ]
        )
        self.rag = RAGAssistant(knowledge_base=self.kb)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model, use_auth_token=hf_token)
        # some tokenizers report huge values for model_max_length, guard it:
        try:
            self.model_max_len = int(self.tokenizer.model_max_length)
        except Exception:
            self.model_max_len = 2048  # safe fallback

        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model, use_auth_token=hf_token, device_map="auto"  # if you want GPU support
        )

        self.generator = pipeline("text2text-generation", model=self.model_obj, device=self.device)

    def _num_tokens(self, text: str) -> int:
        # returns number of tokens for text
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _chunk_strings_by_token_limit(self, strings: list[str], max_tokens: int) -> list[str]:
        """Join strings into chunks each <= max_tokens (token-based, preserves whole strings)."""
        chunks = []
        cur = []
        cur_tokens = 0
        for s in strings:
            toks = self._num_tokens(s)
            if toks > max_tokens:
                # if single string is too long, truncate at char-level conservatively
                # prefer token-level truncation for safety:
                enc = self.tokenizer.encode(s, add_special_tokens=False)
                truncated = self.tokenizer.decode(enc[:max_tokens])
                if cur:
                    chunks.append("\n".join(cur))
                    cur = []
                    cur_tokens = 0
                chunks.append(truncated)
                continue
            if cur_tokens + toks > max_tokens:
                chunks.append("\n".join(cur))
                cur = [s]
                cur_tokens = toks
            else:
                cur.append(s)
                cur_tokens += toks
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def _summarize_chunks(self, chunks: list[str], summary_prompt_prefix: str = "Summarize briefly:") -> str:
        """Summarize each chunk with the generator and then join + summarize the summaries if needed."""
        summaries = []

        for c in chunks:
            prompt = f"{summary_prompt_prefix}\n\n{c}\n\nSummary:"
            out = self.generator(prompt, max_length=64, num_return_sequences=1, clean_up_tokenization_spaces=True)
            summaries.append(out[0]["generated_text"].strip())
        # if multiple summaries, summarize them into one
        if len(summaries) == 1:
            return summaries[0]
        big = "\n".join(summaries)
        prompt = f"Summarize these points into a short single-line summary:\n\n{big}\n\nSummary:"
        out = self.generator(prompt, max_length=64, num_return_sequences=1, clean_up_tokenization_spaces=True)
        return out[0]["generated_text"].strip()

    def _ensure_prompt_fits(
        self,
        system_prompt: str,
        memory_context: str,
        rag_context: str,
        user_message: str,
        reserved_output_tokens: int = 64,
    ) -> str:
        """
        Ensure final prompt fits into model window. Steps:
        1) check length
        2) force memory summarization (chunked) if needed
        3) summarize RAG docs if needed
        4) final truncation preserving system prompt and recent turns
        """

        # Build candidate prompt
        def build_prompt(mem, rag):
            return (
                f"System: {system_prompt}\n"
                f"{mem}\n"
                f"{rag}\n\n"
                f"User: {user_message}\n"
                f"Assistant: Please answer clearly and naturally using the information above."
            )

        prompt = build_prompt(memory_context, rag_context)
        allowed_input = max(1, self.model_max_len - reserved_output_tokens)

        if self._num_tokens(prompt) <= allowed_input:
            return prompt

        # 1) Try compressing memory: chunk old turns -> summarize each chunk
        # We assume memory_context is produced by MemoryManager.build_context() as lines
        mem_lines = memory_context.splitlines()
        if mem_lines:
            chunks = self._chunk_strings_by_token_limit(mem_lines, max_tokens=max(64, allowed_input // 4))
            compressed_mem = self._summarize_chunks(chunks)
            prompt = build_prompt(compressed_mem, rag_context)
            if self._num_tokens(prompt) <= allowed_input:
                return prompt

        # 2) Try summarizing retrieved docs (short bullets)
        if rag_context:
            # rag_context was like "[Relevant company information:]\n- doc1\n- doc2..."
            docs = [line.strip("- ").strip() for line in rag_context.splitlines() if line.startswith("- ")]
            if docs:
                doc_chunks = self._chunk_strings_by_token_limit(docs, max_tokens=max(64, allowed_input // 6))
                short_docs_summary = self._summarize_chunks(
                    doc_chunks, summary_prompt_prefix="Summarize this document into one short bullet:"
                )
                short_rag = f"[Relevant company information summary:]\n- {short_docs_summary}"
                prompt = build_prompt(compressed_mem if "compressed_mem" in locals() else memory_context, short_rag)
                if self._num_tokens(prompt) <= allowed_input:
                    return prompt

        # 3) Last resort: keep system prompt + memory summary + last few user/assistant turns (tail) + truncated rag
        # Preserve the most recent N tokens
        # Keep system prompt intact, then keep last tokens from memory + user
        tail_keep_tokens = allowed_input - self._num_tokens(f"System: {system_prompt}\n\n")
        # create combined tail from memory_context + user_message + rag_context
        combined_tail = "\n".join([memory_context, rag_context, f"User: {user_message}"])
        enc = self.tokenizer.encode(combined_tail, add_special_tokens=False)
        truncated_enc = enc[-tail_keep_tokens:]  # keep tail
        truncated_tail = self.tokenizer.decode(truncated_enc, clean_up_tokenization_spaces=True)
        final_prompt = f"System: {system_prompt}\n{truncated_tail}\n\nAssistant: Please answer clearly and naturally using the information above."
        return final_prompt

    def text_to_text(self, message: str, **generate_kwargs) -> str:
        # Add user turn
        self.memory.add_turn("User", message)

        # If too long, force summarize
        if len(self.memory.turns) > 0 and self._num_tokens(self.memory.build_context()) > (self.max_length // 2):
            # self.memory.force_summarize(generator=self.generator)
            self.memory.force_summarize()

        system_prompt = "You are a polite and professional virtual phone assistant for ACME Corp."
        memory_context = self.memory.build_context()
        rag_context = self.rag.retrieve(message)

        prompt = self._ensure_prompt_fits(
            system_prompt, memory_context, rag_context, message, reserved_output_tokens=64
        )
        logger.debug(prompt)
        outputs = self.generator(
            prompt,
            max_length=self.max_length,
            num_return_sequences=self.num_return_sequences,
            clean_up_tokenization_spaces=True,
            **generate_kwargs,
        )
        reply = outputs[0]["generated_text"]

        # Add assistant turn
        self.memory.add_turn("Assistant", reply)

        return reply
