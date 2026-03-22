from src.internship_fit_analyzer.analyzer import analyze_fit


def test_analyze_fit_returns_missing_skill_and_score_range() -> None:
    resume = """
    Python, pandas, scikit-learn, Git, SQL, and Streamlit projects.
    Built a churn prediction dashboard and worked on APIs with Flask.
    """
    job = """
    Looking for a machine learning intern with Python, pandas, scikit-learn, SQL,
    Git, NLP, deployment, and dashboard experience.
    """

    result = analyze_fit(resume, job)

    assert 0 <= result.overall_score <= 100
    assert "nlp" in result.missing_skills
    assert "python" in result.matched_skills
