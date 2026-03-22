"""Core NLP pipeline for internship fit analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .skill_data import ROLE_PROFILES, SKILL_CATALOG


STOP_WORDS = set(ENGLISH_STOP_WORDS)


@dataclass
class AnalysisResult:
    overall_score: int
    semantic_similarity: int
    skill_match_score: int
    coverage_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    bonus_skills: List[str]
    extracted_resume_skills: Dict[str, List[str]]
    extracted_job_skills: Dict[str, List[str]]
    strengths: List[str]
    recommendations: List[str]
    interview_questions: List[str]
    project_ideas: List[str]
    learning_roadmap: List[str]


def normalize_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9+#./\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_keywords(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]


def extract_skills(text: str) -> Dict[str, List[str]]:
    normalized = normalize_text(text)
    found: Dict[str, List[str]] = {}
    for category, phrases in SKILL_CATALOG.items():
        matches = [phrase for phrase in phrases if phrase in normalized]
        if matches:
            found[category] = sorted(set(matches))
    return found


def flatten_skills(skill_map: Dict[str, List[str]]) -> List[str]:
    flattened = []
    for skills in skill_map.values():
        flattened.extend(skills)
    return sorted(set(flattened))


def semantic_similarity_score(resume_text: str, job_text: str) -> int:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
    return int(round(similarity * 100))


def role_alignment(extracted_job_skills: Dict[str, List[str]]) -> Tuple[str, Dict[str, List[str]]]:
    best_role = "ML Intern"
    best_score = -1

    for role, profile in ROLE_PROFILES.items():
        score = 0
        for category in profile["must_have"]:
            if category in extracted_job_skills:
                score += 2
        for category in profile["nice_to_have"]:
            if category in extracted_job_skills:
                score += 1
        if score > best_score:
            best_role = role
            best_score = score

    return best_role, ROLE_PROFILES[best_role]


def build_strengths(
    matched_skills: List[str], semantic_similarity: int, bonus_skills: List[str]
) -> List[str]:
    strengths = []
    if semantic_similarity >= 55:
        strengths.append("Your resume language aligns well with the internship description.")
    if len(matched_skills) >= 6:
        strengths.append("You already cover a solid set of technical requirements for this role.")
    if bonus_skills:
        strengths.append(
            f"You have extra supporting skills that can differentiate you: {', '.join(bonus_skills[:4])}."
        )
    if not strengths:
        strengths.append("You have a foundation to build on, but the application needs stronger role-specific alignment.")
    return strengths


def build_recommendations(
    missing_skills: List[str], role_name: str, role_profile: Dict[str, List[str]]
) -> Tuple[List[str], List[str], List[str]]:
    recommendations = []
    learning_roadmap = []
    interview_questions = []

    top_missing = missing_skills[:5]
    if top_missing:
        recommendations.append(
            f"Add proof of these missing skills through projects, bullets, or coursework: {', '.join(top_missing)}."
        )
        recommendations.append(
            "Mirror the internship description wording in your resume where it truthfully matches your work."
        )
        learning_roadmap.extend(
            [f"Week {index + 1}: learn and practice {skill} with one mini-deliverable." for index, skill in enumerate(top_missing[:3])]
        )
    else:
        recommendations.append("Your skill coverage is strong, so focus on sharper impact statements and deployment proof.")

    recommendations.append(
        f"Tailor at least one project specifically for a {role_name} position and quantify the result."
    )
    recommendations.extend(role_profile["projects"][:2])
    learning_roadmap.extend(
        [f"Complete or revise one module on {course}." for course in role_profile["courses"][:2]]
    )

    interview_questions = [
        "Which features from your project most influenced the final prediction, and why?",
        "How did you choose your evaluation metric for this internship-style project?",
        "What tradeoff would you make if the model became accurate but too slow for production?",
    ]
    if top_missing:
        interview_questions.append(
            f"How would you quickly get comfortable with {top_missing[0]} before the internship starts?"
        )

    return recommendations, learning_roadmap, interview_questions


def analyze_fit(resume_text: str, job_text: str) -> AnalysisResult:
    normalized_resume = normalize_text(resume_text)
    normalized_job = normalize_text(job_text)

    resume_skills = extract_skills(normalized_resume)
    job_skills = extract_skills(normalized_job)

    resume_skill_list = flatten_skills(resume_skills)
    job_skill_list = flatten_skills(job_skills)

    matched = sorted(set(resume_skill_list) & set(job_skill_list))
    missing = sorted(set(job_skill_list) - set(resume_skill_list))
    bonus = sorted(set(resume_skill_list) - set(job_skill_list))

    semantic_similarity = semantic_similarity_score(normalized_resume, normalized_job)
    skill_match_score = (
        int(round((len(matched) / len(job_skill_list)) * 100)) if job_skill_list else 0
    )

    resume_keywords = set(extract_keywords(normalized_resume))
    job_keywords = set(extract_keywords(normalized_job))
    keyword_overlap = len(resume_keywords & job_keywords)
    keyword_total = max(len(job_keywords), 1)
    coverage_score = int(round((keyword_overlap / keyword_total) * 100))

    overall_score = int(
        round(
            (semantic_similarity * 0.35)
            + (skill_match_score * 0.45)
            + (coverage_score * 0.20)
        )
    )
    overall_score = max(0, min(overall_score, 100))

    role_name, role_profile = role_alignment(job_skills)
    strengths = build_strengths(matched, semantic_similarity, bonus)
    recommendations, learning_roadmap, interview_questions = build_recommendations(
        missing, role_name, role_profile
    )

    return AnalysisResult(
        overall_score=overall_score,
        semantic_similarity=semantic_similarity,
        skill_match_score=skill_match_score,
        coverage_score=coverage_score,
        matched_skills=matched,
        missing_skills=missing,
        bonus_skills=bonus,
        extracted_resume_skills=resume_skills,
        extracted_job_skills=job_skills,
        strengths=strengths,
        recommendations=recommendations,
        interview_questions=interview_questions,
        project_ideas=role_profile["projects"],
        learning_roadmap=learning_roadmap,
    )
