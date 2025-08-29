import numpy as np


def float_to_pcm16(frame: np.ndarray) -> bytes:
    """Converts float to pcm16.

    Args:
        frame (np.ndarray): _description_

    Returns:
        bytes: _description_

    """
    frame = np.clip(frame, -1.0, 1.0)  # ensure range
    ints = (frame * 32767).astype(np.int16)
    return ints.tobytes()
