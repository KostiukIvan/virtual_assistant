import math

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import pkg.config as config
from pkg.ai.models.stt.stt_interface import SpeechToTextModel


class Whisper(SpeechToTextModel):
    def __init__(self, model: str = config.STT_MODEL) -> None:
        super().__init__(model, config.DEVICE_CUDA_OR_CPU)
        # Load model and processor
        self.device = torch.device(config.DEVICE_CUDA_OR_CPU)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model)
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model)

    def audio_to_text(self, buffer: np.ndarray, sample_rate: int):
        """
        buffer: numpy array of audio samples (1D or maybe 2D if multi-channel)
        sample_rate: sampling rate of buffer

        Returns:
            text: str
            confidence: float between 0.0 and 1.0 (approximate)
        """
        # Preprocess
        input_features = self.processor(buffer, sampling_rate=sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        # Generate with scores
        # Set arguments for generation
        # return_dict_in_generate and output_scores to get token scores
        gen_kwargs = {
            "return_dict_in_generate": True,
            "output_scores": True,
            # optionally adjust temperature, beam size etc
            # e.g. "num_beams": 1,
            # you might want to suppress tokens etc depending on your needs
        }

        outputs = self.model.generate(input_features, **gen_kwargs)
        # outputs is a Seq2SeqGeneratorOutput with attributes:
        #   sequences (token ids), scores, etc

        # Decode text
        text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

        # Compute confidence from output scores
        # outputs.scores is a list of logits (or logit scores) for each generated token position
        # For each token generated, find the log probability of the chosen token, sum or average them

        # First, convert logits to log-probs
        # outputs.scores has shape: list of length T (number of decoding steps),
        # each element is a tensor of shape (batch_size * num_beams, vocab_size) or (batch_size, vocab_size)
        # Here, assume batch_size=1, no beam search (or flatten accordingly)

        # Let's do the following:

        sequences = outputs.sequences  # shape: (batch_size, seq_len)
        # If using beams or multiple hypotheses, you may need to pick the first/hypothesis

        # for each decoding step i, get the score tensor for that step,
        # look up the log-prob of the token that was chosen

        # Note: sometimes outputs.scores[i] corresponds to token i+1 (because the first token might be decoder_start_token etc).
        # Need to align correctly.

        # Simple version:
        seq = sequences[0]  # token ids
        # skip special tokens if present in beginning; adjust alignment
        # Here we assume outputs.scores length matches (seq_len - 1) or so.

        # Example:
        total_logprob = 0.0
        count = 0
        for i, score_logits in enumerate(outputs.scores):
            # score_logits: tensor of shape (batch_size, vocab_size) or (batch_size * num_beams,...)
            # We'll assume no beam search and batch_size=1
            # get the token id at position i+1 in seq
            token_id = seq[i + 1]  # note offset by one; adjust if needed
            # get log-prob: apply log_softmax
            log_probs_i = torch.log_softmax(score_logits[0], dim=-1)
            token_logprob = log_probs_i[token_id]
            total_logprob += token_logprob.item()
            count += 1

        if count > 0:
            avg_logprob = total_logprob / count
            # Option A: sigmoid
            confidence = 1 / (1 + math.exp(-avg_logprob))
            # Option B (alternative): linear rescale
            # confidence = max(0.0, min(1.0, (avg_logprob + 5) / 5))
        else:
            confidence = 0.0

        return text, confidence
