# test_aec_enabled_audio_io_hypothesis.py
import queue
import time

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from pkg.ai.streams.acousting_echo_canceller import AcousticEchoCanceller


def make_aec(frame_size=160):
    mic_q = queue.Queue()
    ref_q = queue.Queue()
    out_q = queue.Queue()
    return AcousticEchoCanceller(
        mic_queue=mic_q,
        playback_ref_queue=ref_q,
        output_queue=out_q,
        sample_rate=16000,
        frame_size=frame_size,
        mu=0.2,
        leak=1e-4,
        delta=10.0,
        epsilon=1e-6,
        energy_threshold=10.0,
    )


# -----------------------------
# PROPERTY-BASED TESTS
# -----------------------------


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=160, max_size=160
    )
)
@settings(deadline=None, max_examples=25)
def test_energy_is_non_negative(signal):
    aec = make_aec()
    arr = np.array(signal, dtype=np.float32).reshape(-1, 1)
    energy = aec._compute_energy(arr)
    assert energy >= 0.0
    assert not np.isnan(energy)
    assert not np.isinf(energy)


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=160, max_size=160
    ),
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=160, max_size=160
    ),
)
@settings(deadline=None, max_examples=10)
def test_aec_pipeline_never_crashes(mic_signal, ref_signal):
    """Ensure pipeline runs without NaNs/Inf for random mic+ref frames."""
    aec = make_aec()
    mic_q, ref_q, out_q = aec.mic_queue, aec.playback_ref_queue, aec.output_queue
    aec.start()

    mic = np.array(mic_signal, dtype=np.float32).reshape(-1, 1)
    ref = np.array(ref_signal, dtype=np.float32).reshape(-1, 1)

    mic_q.put(mic)
    ref_q.put(ref)
    time.sleep(0.05)

    aec.stop()

    while not out_q.empty():
        cleaned = out_q.get()
        assert not np.any(np.isnan(cleaned))
        assert not np.any(np.isinf(cleaned))


@given(
    st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=160, max_size=160
    ),
)
@settings(deadline=None, max_examples=10)
def test_double_talk_hold_freezes_filter(random_signal):
    """If near-end dominates, H should not adapt significantly."""
    aec = make_aec()
    mic_q, ref_q, _ = aec.mic_queue, aec.playback_ref_queue, aec.output_queue
    aec.start()

    ref = np.zeros((aec.frame_size, 1), dtype=np.float32)
    mic = np.array(random_signal, dtype=np.float32).reshape(-1, 1) * 5.0  # very loud near-end

    # Push several frames
    for _ in range(5):
        ref_q.put(ref)
        mic_q.put(mic)
        time.sleep(0.01)

    H_before = aec.H.copy()
    time.sleep(0.05)
    H_after = aec.H.copy()

    aec.stop()

    diff = np.linalg.norm(H_after - H_before)
    assert diff < 1e-2  # minimal adaptation


@given(st.integers(min_value=64, max_value=1024))
@settings(deadline=None, max_examples=10)
def test_aec_respects_frame_size(frame_size):
    """Randomized frame size still produces valid output shapes."""
    aec = make_aec(frame_size=frame_size)
    mic_q, ref_q, out_q = aec.mic_queue, aec.playback_ref_queue, aec.output_queue
    aec.start()

    mic = np.zeros((frame_size, 1), dtype=np.float32)
    ref = np.zeros((frame_size, 1), dtype=np.float32)

    mic_q.put(mic)
    ref_q.put(ref)
    time.sleep(0.05)

    aec.stop()

    if not out_q.empty():
        out = out_q.get()
        assert out.shape == (frame_size, 1)
