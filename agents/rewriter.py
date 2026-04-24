"""
Rewriter Agent
Uses Google Gemini to craft an optimized LinkedIn profile based on the
analysis findings and research context.
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


def _research_summary(research) -> str:
    """Convert the research object (dict or string) to a concise text block."""
    if isinstance(research, dict):
        combined = research.get("combined_text", "")
        keywords = research.get("trending_keywords", [])
        kw_str = ", ".join(keywords[:20]) if keywords else "N/A"
        return f"Trending keywords: {kw_str}\n\nResearch context:\n{combined[:2000]}"
    return str(research)[:2000]


def _analysis_summary(analysis: dict) -> str:
    """Summarise the analysis dict into a concise prompt block."""
    if not isinstance(analysis, dict):
        return str(analysis)[:1000]
    lines = [
        f"Overall score: {analysis.get('overall_score', 'N/A')}/10",
        f"Weaknesses: {', '.join(analysis.get('weaknesses', []))}",
        f"Missing keywords: {', '.join(analysis.get('missing_keywords', []))}",
        f"Priority improvements: {', '.join(analysis.get('priority_improvements', []))}",
        f"Missing skills: {', '.join(analysis.get('missing_skills', []))}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_profile(
    target_job: str,
    headline: str,
    about: str,
    skills: str,
    experience: str,
    analysis: dict,
    research,
    gemini_key: str,
) -> dict:
    """
    Rewrite the LinkedIn profile sections for maximum impact.

    Args:
        target_job:  Target job title.
        headline:    Current LinkedIn headline.
        about:       Current About / Summary section.
        skills:      Comma-separated list of current skills.
        experience:  Key experience bullet points.
        analysis:    Analysis results from the Analyzer agent.
        research:    Research findings from the Researcher agent.
        gemini_key:  Google Gemini API key.

    Returns:
        A dictionary with optimized profile sections.
    """
    research_text = _research_summary(research)
    analysis_text = _analysis_summary(analysis)

    prompt = f"""You are a world-class LinkedIn profile writer and personal branding expert.

Rewrite the LinkedIn profile below to maximize recruiter appeal and ATS compatibility
for the target role of "{target_job}".

--- ANALYSIS FINDINGS ---
{analysis_text}

--- RESEARCH CONTEXT ---
{research_text}

--- CURRENT PROFILE ---
Headline: {headline or "(not provided)"}
About: {about or "(not provided)"}
Skills: {skills or "(not provided)"}
Experience: {experience or "(not provided)"}

Guidelines:
- Headline: punchy, keyword-rich, ≤220 characters, no buzzwords like "guru" or "ninja"
- About: 3-5 short paragraphs, first-person, achievement-focused, include quantified results
  where possible, end with a clear call-to-action
- Skills: exactly 15 skills ordered by relevance to the target role
- featured_keywords: 10-15 ATS-critical keywords to weave into the profile
- recruiter_tip: one concise, actionable tip for the candidate

Return ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:
{{
  "headline": "<optimized headline string>",
  "headline_options": [<2 alternative headline strings>],
  "about": "<optimized about section — use \\n for paragraph breaks>",
  "skills": [<list of exactly 15 skill strings>],
  "featured_keywords": [<list of 10-15 keyword strings>],
  "recruiter_tip": "<one actionable tip string>"
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
            "headline": headline or f"{target_job} | Open to Opportunities",
            "headline_options": [],
            "about": about or "",
            "skills": [],
            "featured_keywords": [],
            "recruiter_tip": "Keep your profile updated and engage with industry content regularly.",
        }
        for key, default in defaults.items():
            result.setdefault(key, default)

        return result

    except json.JSONDecodeError:
        return {
            "headline": headline or f"{target_job} | Open to Opportunities",
            "headline_options": [],
            "about": about or "",
            "skills": [s.strip() for s in skills.split(",") if s.strip()] if skills else [],
            "featured_keywords": [],
            "recruiter_tip": "Re-run the optimizer to get a fully rewritten profile.",
            "_raw_response": response.text if "response" in dir() else "",
        }
    except Exception as e:
        raise RuntimeError(f"Rewriter agent failed: {e}") from e
