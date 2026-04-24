"""
Analyzer Agent
Uses Google Gemini to score the user's current LinkedIn profile against
the target job role and research findings, returning structured feedback.
"""

import json
import re
from google import genai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_json(text: str) -> str:
    """Strip markdown code fences that Gemini sometimes wraps around JSON."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _research_summary(research) -> str:
    """Convert the research object (dict or string) to a concise text block."""
    if isinstance(research, dict):
        combined = research.get("combined_text", "")
        keywords = research.get("trending_keywords", [])
        kw_str = ", ".join(keywords[:20]) if keywords else "N/A"
        return f"Trending keywords: {kw_str}\n\nResearch context:\n{combined[:2000]}"
    return str(research)[:2000]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_profile(
    target_job: str,
    headline: str,
    about: str,
    skills: str,
    experience: str,
    research,
    gemini_key: str,
) -> dict:
    """
    Analyze the user's LinkedIn profile against the target job role.

    Args:
        target_job:  Target job title.
        headline:    Current LinkedIn headline.
        about:       Current About / Summary section.
        skills:      Comma-separated list of current skills.
        experience:  Key experience bullet points.
        research:    Research findings from the Researcher agent (dict or str).
        gemini_key:  Google Gemini API key.

    Returns:
        A dictionary with scores and detailed feedback.
    """
    research_text = _research_summary(research)

    prompt = f"""You are an expert LinkedIn profile analyst and career coach.

Analyze the LinkedIn profile below for the target role of "{target_job}".
Use the research context to identify gaps and opportunities.

--- RESEARCH CONTEXT ---
{research_text}

--- CURRENT PROFILE ---
Headline: {headline or "(not provided)"}
About: {about or "(not provided)"}
Skills: {skills or "(not provided)"}
Experience: {experience or "(not provided)"}

Return ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
{{
  "overall_score": <integer 1-10>,
  "headline_score": <integer 1-10>,
  "about_score": <integer 1-10>,
  "keyword_score": <integer 1-10>,
  "skills_score": <integer 1-10>,
  "ats_compatibility": <integer 1-10>,
  "strengths": [<up to 5 short strings>],
  "weaknesses": [<up to 5 short strings>],
  "missing_keywords": [<up to 10 keyword strings>],
  "priority_improvements": [<up to 5 actionable strings>],
  "missing_skills": [<up to 8 skill strings>],
  "missing_certifications": [<up to 5 certification strings>]
}}"""

    try:
        client = genai.Client(api_key=gemini_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = _clean_json(response.text)
        result = json.loads(raw)

        # Ensure all expected keys are present with safe defaults
        defaults = {
            "overall_score": 5,
            "headline_score": 5,
            "about_score": 5,
            "keyword_score": 5,
            "skills_score": 5,
            "ats_compatibility": 5,
            "strengths": [],
            "weaknesses": [],
            "missing_keywords": [],
            "priority_improvements": [],
            "missing_skills": [],
            "missing_certifications": [],
        }
        for key, default in defaults.items():
            result.setdefault(key, default)

        return result

    except json.JSONDecodeError:
        # Gemini returned non-JSON — return a safe fallback with the raw text
        return {
            "overall_score": 5,
            "headline_score": 5,
            "about_score": 5,
            "keyword_score": 5,
            "skills_score": 5,
            "ats_compatibility": 5,
            "strengths": ["Profile received"],
            "weaknesses": ["Could not parse detailed analysis"],
            "missing_keywords": [],
            "priority_improvements": ["Re-run the optimizer to get a full analysis"],
            "missing_skills": [],
            "missing_certifications": [],
            "_raw_response": response.text if "response" in dir() else "",
        }
    except Exception as e:
        raise RuntimeError(f"Analyzer agent failed: {e}") from e
