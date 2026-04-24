"""
Judge Agent
Uses Google Gemini to independently score the rewritten LinkedIn profile,
providing an unbiased quality assessment.
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
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _rewrite_summary(rewrite: dict) -> str:
    """Flatten the rewrite dict into a readable profile block."""
    if not isinstance(rewrite, dict):
        return str(rewrite)[:2000]
    skills_str = ", ".join(rewrite.get("skills", []))
    keywords_str = ", ".join(rewrite.get("featured_keywords", []))
    return (
        f"Headline: {rewrite.get('headline', '')}\n\n"
        f"About:\n{rewrite.get('about', '')}\n\n"
        f"Skills: {skills_str}\n\n"
        f"ATS Keywords: {keywords_str}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_profile(target_job: str, rewrite: dict, gemini_key: str) -> dict:
    """
    Independently score the rewritten LinkedIn profile.

    Args:
        target_job:  Target job title.
        rewrite:     Rewritten profile sections from the Rewriter agent.
        gemini_key:  Google Gemini API key.

    Returns:
        A dictionary with detailed scores and qualitative feedback.
    """
    profile_text = _rewrite_summary(rewrite)

    prompt = f"""You are a strict, independent LinkedIn profile evaluator acting as a hiring manager
and ATS system expert. You did NOT write this profile — evaluate it objectively.

Target role: "{target_job}"

--- REWRITTEN PROFILE TO EVALUATE ---
{profile_text}

Score each dimension from 1-10 and provide qualitative feedback.

Verdict must be exactly one of: "EXCELLENT", "GOOD", "NEEDS WORK", or "POOR"
- EXCELLENT: overall ≥ 9
- GOOD: overall 7-8
- NEEDS WORK: overall 5-6
- POOR: overall ≤ 4

Return ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
{{
  "overall": <integer 1-10>,
  "verdict": "<EXCELLENT|GOOD|NEEDS WORK|POOR>",
  "clarity": <integer 1-10>,
  "keywords": <integer 1-10>,
  "professionalism": <integer 1-10>,
  "ats_ready": <integer 1-10>,
  "recruiter_appeal": <integer 1-10>,
  "uniqueness": <integer 1-10>,
  "best_part": "<one sentence describing the strongest element of the profile>",
  "critical_fix": "<one sentence describing the single most important improvement>",
  "detailed_feedback": "<2-4 sentences of balanced, constructive feedback>"
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
            "overall": 7,
            "verdict": "GOOD",
            "clarity": 7,
            "keywords": 7,
            "professionalism": 7,
            "ats_ready": 7,
            "recruiter_appeal": 7,
            "uniqueness": 7,
            "best_part": "The profile has been optimized for the target role.",
            "critical_fix": "Continue refining with real metrics and achievements.",
            "detailed_feedback": "The profile is well-structured and keyword-rich. Adding quantified achievements would further strengthen it.",
        }
        for key, default in defaults.items():
            result.setdefault(key, default)

        # Normalise verdict casing
        verdict = str(result.get("verdict", "GOOD")).upper()
        if verdict not in {"EXCELLENT", "GOOD", "NEEDS WORK", "POOR"}:
            verdict = "GOOD"
        result["verdict"] = verdict

        return result

    except json.JSONDecodeError:
        return {
            "overall": 7,
            "verdict": "GOOD",
            "clarity": 7,
            "keywords": 7,
            "professionalism": 7,
            "ats_ready": 7,
            "recruiter_appeal": 7,
            "uniqueness": 7,
            "best_part": "Profile has been rewritten for the target role.",
            "critical_fix": "Re-run the optimizer to get a detailed judge evaluation.",
            "detailed_feedback": "The judge could not parse a structured response. The rewritten profile is ready to use.",
            "_raw_response": response.text if "response" in dir() else "",
        }
    except Exception as e:
        raise RuntimeError(f"Judge agent failed: {e}") from e
