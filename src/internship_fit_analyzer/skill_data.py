"""Curated skill taxonomy and role roadmaps for internship analysis."""

SKILL_CATALOG = {
    "python": ["python", "pandas", "numpy", "matplotlib", "seaborn", "jupyter"],
    "machine_learning": [
        "machine learning",
        "scikit-learn",
        "sklearn",
        "regression",
        "classification",
        "clustering",
        "model evaluation",
        "feature engineering",
    ],
    "deep_learning": [
        "deep learning",
        "tensorflow",
        "pytorch",
        "cnn",
        "rnn",
        "lstm",
        "transformers",
    ],
    "nlp": [
        "nlp",
        "natural language processing",
        "text classification",
        "named entity recognition",
        "sentiment analysis",
        "bert",
    ],
    "data_analysis": [
        "sql",
        "excel",
        "statistics",
        "hypothesis testing",
        "data analysis",
        "data visualization",
    ],
    "backend": [
        "flask",
        "fastapi",
        "django",
        "rest api",
        "apis",
        "microservices",
    ],
    "frontend": [
        "html",
        "css",
        "javascript",
        "react",
        "streamlit",
        "ui",
    ],
    "cloud": [
        "aws",
        "gcp",
        "azure",
        "docker",
        "kubernetes",
        "deployment",
        "ci/cd",
    ],
    "tools": [
        "git",
        "github",
        "linux",
        "bash",
        "debugging",
        "testing",
    ],
    "soft_skills": [
        "communication",
        "leadership",
        "collaboration",
        "problem solving",
        "ownership",
    ],
}


ROLE_PROFILES = {
    "ML Intern": {
        "must_have": ["python", "machine_learning", "data_analysis", "tools"],
        "nice_to_have": ["nlp", "deep_learning", "backend", "cloud", "soft_skills"],
        "projects": [
            "Build an end-to-end ML project with data cleaning, model comparison, and deployment.",
            "Add an explainability section that highlights which features influence predictions.",
            "Publish a polished README with dataset, metrics, screenshots, and interview-ready talking points.",
        ],
        "courses": [
            "Scikit-learn model building and evaluation",
            "SQL for analytics",
            "Intro to deployment with Streamlit or FastAPI",
        ],
    },
    "Data Science Intern": {
        "must_have": ["python", "data_analysis", "machine_learning", "tools"],
        "nice_to_have": ["cloud", "soft_skills", "backend"],
        "projects": [
            "Create a dashboard-driven analytics project with forecasting or segmentation.",
            "Run an A/B-test style analysis on a public dataset and explain business impact.",
            "Document assumptions and data quality checks like a real analyst.",
        ],
        "courses": [
            "Statistics and experimentation",
            "Advanced pandas and visualization",
            "Business metrics and storytelling",
        ],
    },
    "Software + AI Intern": {
        "must_have": ["python", "backend", "tools", "frontend"],
        "nice_to_have": ["machine_learning", "cloud", "soft_skills"],
        "projects": [
            "Build a full-stack AI feature with a clean UI and measurable output.",
            "Expose model predictions through an API and log failed inputs for debugging.",
            "Show code quality with tests, modular structure, and deployment instructions.",
        ],
        "courses": [
            "APIs with FastAPI or Flask",
            "Frontend basics for technical demos",
            "Docker fundamentals",
        ],
    },
}
