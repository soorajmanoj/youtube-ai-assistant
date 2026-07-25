from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import src.main as main_module

client = TestClient(main_module.app)


def setup_function():
    # Clear sessions between tests so they don't leak into each other.
    main_module.sessions.clear()


def test_process_video_success(monkeypatch):
    monkeypatch.setattr(main_module, "get_transcript", lambda url: "some transcript text")
    monkeypatch.setattr(main_module, "build_vectorstore", lambda text: MagicMock())
    monkeypatch.setattr(main_module, "create_qa_chain", lambda vs: MagicMock())

    res = client.post("/process", json={"url": "https://www.youtube.com/watch?v=abc123"})

    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "session_id" in body
    assert body["session_id"] in main_module.sessions


def test_process_video_empty_url():
    res = client.post("/process", json={"url": "   "})
    assert res.status_code == 400


def test_process_video_invalid_url(monkeypatch):
    def raise_invalid(url):
        raise ValueError("Invalid YouTube URL.")

    monkeypatch.setattr(main_module, "get_transcript", raise_invalid)

    res = client.post("/process", json={"url": "not-a-youtube-url"})
    assert res.status_code == 400
    assert "Invalid YouTube URL" in res.json()["detail"]


def test_process_video_ollama_down(monkeypatch):
    monkeypatch.setattr(main_module, "get_transcript", lambda url: "transcript")

    def raise_connection_error(text):
        raise Exception("Connection refused")

    monkeypatch.setattr(main_module, "build_vectorstore", raise_connection_error)

    res = client.post("/process", json={"url": "https://www.youtube.com/watch?v=abc123"})
    assert res.status_code == 503


def test_ask_without_session():
    res = client.post("/ask", json={"session_id": "does-not-exist", "question": "What happens?"})
    assert res.status_code == 404


def test_ask_empty_question():
    res = client.post("/ask", json={"session_id": "whatever", "question": "  "})
    assert res.status_code == 400


def test_ask_success(monkeypatch):
    monkeypatch.setattr(main_module, "get_transcript", lambda url: "some transcript text")
    monkeypatch.setattr(main_module, "build_vectorstore", lambda text: MagicMock())

    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {"answer": "42"}
    monkeypatch.setattr(main_module, "create_qa_chain", lambda vs: fake_chain)

    process_res = client.post("/process", json={"url": "https://www.youtube.com/watch?v=abc123"})
    session_id = process_res.json()["session_id"]

    ask_res = client.post("/ask", json={"session_id": session_id, "question": "What is the answer?"})
    assert ask_res.status_code == 200
    assert ask_res.json()["answer"] == "42"