class MemoryManager:
    def __init__(
        self,
        ttt_model,  # any model with `text_to_text()`
        window_size: int = 6,
        summarize_every: int = 20,
    ):
        """
        Args:
            ttt_model: Your LocalTextToTextModel (or similar).
            window_size: How many recent turns to keep verbatim.
            summarize_every: After how many turns we trigger summarization.
        """
        self.ttt_model = ttt_model
        self.window_size = window_size
        self.summarize_every = summarize_every
        self.turns: list[tuple[str, str]] = []  # (role, text)
        self.summary: str = ""

    def add_turn(self, role: str, text: str):
        """Add a new conversation turn."""
        self.turns.append((role, text))

        # trigger summarization if needed
        if len(self.turns) > self.summarize_every:
            self._summarize_history()

    def _summarize_history(self):
        """Use TTT model to summarize older history into compact notes."""
        old_turns = self.turns[: -self.window_size]

        if old_turns:
            history_text = "\n".join([f"{r}: {t}" for r, t in old_turns])

            # Ask the TTT model to summarize
            prompt = (
                "Summarize the following conversation briefly, keeping key facts, "
                "decisions, and user intents:\n\n"
                f"{history_text}\n\nSummary:"
            )

            try:
                compact = self.ttt_model.text_to_text(prompt, max_length=128)
            except Exception:
                # Fallback: naive concatenation
                compact = " ".join([t for _, t in old_turns])

            if self.summary:
                self.summary += " " + compact
            else:
                self.summary = compact

        # keep only the recent window
        self.turns = self.turns[-self.window_size :]

    def build_prompt(self, system_prompt: str | None = None) -> str:
        """Return full conversation context (summary + recent turns)."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if self.summary:
            parts.append(f"[Summary of earlier conversation: {self.summary}]")
        for role, text in self.turns:
            parts.append(f"{role}: {text}")
        parts.append("Assistant:")  # cue for generation
        return "\n".join(parts)
