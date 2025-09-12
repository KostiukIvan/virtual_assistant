class MemoryManager:
    def __init__(self, window_size: int = 6, summarize_every: int = 20):
        self.window_size = window_size
        self.summarize_every = summarize_every
        self.turns: list[tuple[str, str]] = []
        self.summaries: list[str] = []

    def add_turn(self, role: str, text: str):
        self.turns.append((role, text))

        # auto-summarize every N turns
        if len(self.turns) >= self.summarize_every:
            self.force_summarize()

    def build_context(self) -> str:
        """
        Returns conversation history for prompt building.
        Includes accumulated summaries + last window_size turns.
        """
        parts = []

        if self.summaries:
            parts.append("Previous summary:\n" + "\n".join(self.summaries))

        # take only the last N turns
        for role, text in self.turns[-self.window_size :]:
            parts.append(f"{role}: {text}")

        return "\n".join(parts)

    def force_summarize(self, generator=None, max_length: int = 80):
        """
        Summarize the current conversation history into a shorter form.
        - generator: HuggingFace pipeline for text2text-generation
        - max_length: max tokens for summary
        """
        if not self.turns:
            return

        # Combine all turns into one block of text
        convo = "\n".join([f"{r}: {t}" for r, t in self.turns])

        if generator is None:
            # Just compress manually (naive fallback)
            summary = f"[Summary of {len(self.turns)} turns]"
        else:
            # Use your TTT model (text2text-generation pipeline)
            prompt = (
                "Summarize this conversation briefly, keeping key facts, names, and decisions:\n\n"
                f"{convo}\n\nSummary:"
            )
            outputs = generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                clean_up_tokenization_spaces=True,
            )
            summary = outputs[0]["generated_text"].strip()

        # Store summary + clear old turns
        self.summaries.append(summary)
        self.turns = []
