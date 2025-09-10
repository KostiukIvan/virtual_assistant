# test_aec_enabled_audio_io.py
import queue
import time

import numpy as np
import pytest
from pkg.ai.streams.acousting_echo_canceller import AcousticEchoCanceller


@pytest.fixture
def aec():
    mic_q = queue.Queue()
    ref_q = queue.Queue()
    out_q = queue.Queue()
    return AcousticEchoCanceller(
        mic_queue=mic_q,
        playback_ref_queue=ref_q,
        output_queue=out_q,
        sample_rate=16000,
        frame_size=160,  # 10 ms for faster test
        mu=0.2,
        leak=1e-4,
        delta=10.0,
        epsilon=1e-6,
        energy_threshold=10.0,
    )


def test_compute_energy_basic(aec):
    silence = np.zeros((160, 1), dtype=np.float32)
    tone = np.ones((160, 1), dtype=np.float32)
    assert aec._compute_energy(silence) == pytest.approx(0.0, abs=1e-6)
    assert aec._compute_energy(tone) > 0.5
    assert aec._compute_energy(tone[:, 0]) > 0.5  # 1D array also works


def test_aec_reduces_echo(aec):
    mic_q, ref_q, out_q = aec.mic_queue, aec.playback_ref_queue, aec.output_queue
    aec.start()

    fs = aec.sample_rate
    t = np.arange(aec.frame_size) / fs
    freq = 440.0
    ref = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32).reshape(-1, 1)

    # Simulated mic = ref (echo) + small noise
    mic = ref + 0.01 * np.random.randn(*ref.shape).astype(np.float32)

    # push a few frames
    for _ in range(20):
        ref_q.put(ref)
        mic_q.put(mic)
        time.sleep(0.01)

    # Collect some output
    outputs = []
    for _ in range(10):
        try:
            outputs.append(out_q.get(timeout=0.5))
        except queue.Empty:
            pass

    aec.stop()

    assert len(outputs) > 0
    # Compute correlation of mic vs ref, and cleaned vs ref
    mic_corr = np.corrcoef(mic[:, 0], ref[:, 0])[0, 1]
    clean_corr = np.corrcoef(outputs[-1][:, 0], ref[:, 0])[0, 1]
    assert abs(clean_corr) < abs(mic_corr)  # cleaned should correlate less with echo


def test_double_talk_detection_freezes_filter(aec):
    mic_q, ref_q, _out_q = aec.mic_queue, aec.playback_ref_queue, aec.output_queue
    aec.start()

    # Strong near-end (large mic, tiny ref)
    ref = np.zeros((aec.frame_size, 1), dtype=np.float32)
    mic = np.random.randn(aec.frame_size, 1).astype(np.float32)

    # Push several frames
    for _ in range(5):
        ref_q.put(ref)
        mic_q.put(mic)
        time.sleep(0.01)

    # Copy filter after freeze
    H_before = aec.H.copy()
    time.sleep(0.05)
    H_after = aec.H.copy()
    aec.stop()

    # Expect minimal change
    diff = np.linalg.norm(H_after - H_before)
    assert diff < 1e-3


def test_start_and_stop_threads(aec):
    assert not aec.is_running
    aec.start()
    assert aec.is_running
    time.sleep(0.05)
    aec.stop()
    assert not aec.is_running


def test_output_queue_overflow_handling(aec):
    # Create tiny output queue
    aec.output_queue = queue.Queue(maxsize=1)
    aec.start()

    ref = np.zeros((aec.frame_size, 1), dtype=np.float32)
    mic = np.zeros((aec.frame_size, 1), dtype=np.float32)

    # Push more frames than queue can hold
    for _ in range(10):
        aec.playback_ref_queue.put(ref)
        aec.mic_queue.put(mic)
        time.sleep(0.01)

    # If code is correct, it shouldn't crash despite queue full
    aec.stop()
