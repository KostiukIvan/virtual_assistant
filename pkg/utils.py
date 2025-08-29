import numpy as np


def float_to_pcm16(frame: np.ndarray) -> bytes:
    frame = np.clip(frame, -1.0, 1.0)  # ensure range
    ints = (frame * 32767).astype(np.int16)
    return ints.tobytes()
