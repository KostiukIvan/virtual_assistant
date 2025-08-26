import pytest
import asyncio
from unittest.mock import AsyncMock
from pkg.conversational_ai import ConversationalAI  # adjust import if needed

DUMMY_MODEL = "dummy-model"

@pytest.mark.asyncio
async def test_initialization():
    ai = ConversationalAI(model_name=DUMMY_MODEL)
    assert ai.model_name == DUMMY_MODEL
    assert ai.system_prompt == "You are a helpful assistant."
    assert ai.max_context == 10
    assert ai.conversation == []
    assert isinstance(ai.input_queue, asyncio.Queue)
    assert isinstance(ai.output_queue, asyncio.Queue)
    assert ai.processing_task is None

@pytest.mark.asyncio
async def test_enqueue_input_and_response():
    ai = ConversationalAI(model_name=DUMMY_MODEL)
    ai.start()

    # Send input
    await ai.enqueue_input("Hello AI")

    # Wait a little for processing
    response = await ai.get_response(timeout=0.2)
    assert response == "AI reply to: Hello AI"

    # Conversation should include both user and assistant
    assert ai.conversation[-2]["role"] == "user"
    assert ai.conversation[-2]["content"] == "Hello AI"
    assert ai.conversation[-1]["role"] == "assistant"
    assert ai.conversation[-1]["content"] == response

    await ai.stop()

@pytest.mark.asyncio
async def test_multiple_inputs_preserve_context():
    ai = ConversationalAI(model_name=DUMMY_MODEL, max_context=3)
    ai.start()

    # Send multiple messages
    inputs = ["Hi", "How are you?", "Tell me a joke"]
    for msg in inputs:
        await ai.enqueue_input(msg)

    # Collect responses
    responses = []
    for _ in inputs:
        resp = await ai.get_response(timeout=0.3)
        responses.append(resp)

    # Check responses correspond to inputs
    for msg, resp in zip(inputs, responses):
        assert resp == f"AI reply to: {msg}"

    # Conversation length should not exceed max_context
    assert len(ai.conversation) <= ai.max_context

    await ai.stop()

@pytest.mark.asyncio
async def test_stop_signal_stops_processing():
    ai = ConversationalAI(model_name=DUMMY_MODEL)
    ai.start()

    # Stop immediately
    await ai.stop()
    assert ai.processing_task.done()

@pytest.mark.asyncio
async def test_build_prompt_includes_system_prompt_and_messages():
    ai = ConversationalAI(model_name=DUMMY_MODEL, system_prompt="SYS PROMPT")
    ai.conversation = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    prompt = ai._build_prompt()
    expected = "SYS PROMPT\nUser: Hi\nAssistant: Hello"
    assert prompt == expected

@pytest.mark.asyncio
async def test_call_model_returns_expected_response():
    ai = ConversationalAI(model_name=DUMMY_MODEL)
    ai.conversation.append({"role": "user", "content": "Test"})
    response = await ai._call_model(ai._build_prompt())
    assert response == "AI reply to: Test"
