from groq_multi_agent.tools.summarize import summarize_text


def test_summarize_text_truncates_sentences():
    text = "One. Two. Three. Four. Five. Six."
    out = summarize_text.invoke({"text": text, "max_sentences": 3})
    assert "One." in out and "Three." in out
    assert "[...]" in out
    assert "Six." not in out


def test_summarize_empty():
    assert "(empty input)" in summarize_text.invoke({"text": "   "})
