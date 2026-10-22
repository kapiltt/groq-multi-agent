from groq_multi_agent.triage import assess_task, triage_heuristic


def test_rejects_greeting():
    r = triage_heuristic("Hi")
    assert r and not r.eligible


def test_rejects_time():
    r = triage_heuristic("what is current time?")
    assert r and not r.eligible


def test_rejects_name():
    r = triage_heuristic("what is my name?")
    assert r and not r.eligible


def test_accepts_research_task():
    r = assess_task("Compare FastAPI vs Flask for a SaaS API")
    assert r.eligible