"""
Researcher Agent
Uses the Tavily API to search for LinkedIn trends and best practices
for a given job role, returning structured research findings.
"""

from tavily import TavilyClient


def research_job_role(job_title: str, tavily_key: str) -> dict:
    """
    Search for LinkedIn profile trends and best practices for the given job role.

    Args:
        job_title: The target job title to research (e.g. "Senior Embedded Systems Engineer").
        tavily_key: Tavily API key.

    Returns:
        A dictionary containing research findings about the job role.
    """
    try:
        client = TavilyClient(api_key=tavily_key)

        queries = [
            f"LinkedIn profile best practices for {job_title} 2024",
            f"top skills keywords for {job_title} LinkedIn recruiter",
            f"LinkedIn headline examples {job_title}",
        ]

        all_results = []
        for query in queries:
            try:
                response = client.search(
                    query=query,
                    search_depth="basic",
                    max_results=3,
                )
                results = response.get("results", [])
                for r in results:
                    all_results.append({
                        "title": r.get("title", ""),
                        "content": r.get("content", ""),
                        "url": r.get("url", ""),
                    })
            except Exception:
                continue

        # Aggregate content snippets for downstream agents
        snippets = [r["content"] for r in all_results if r.get("content")]
        combined_text = "\n\n".join(snippets[:9])  # cap at 9 snippets

        # Extract a simple keyword list from titles + content
        import re
        raw_text = " ".join(r["title"] + " " + r["content"] for r in all_results)
        # Pull capitalised / technical-looking tokens as candidate keywords
        keyword_candidates = re.findall(r"\b[A-Z][A-Za-z0-9+#./-]{2,}\b", raw_text)
        # Deduplicate while preserving order
        seen: set = set()
        keywords: list = []
        for kw in keyword_candidates:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
        keywords = keywords[:30]

        return {
            "job_title": job_title,
            "raw_results": all_results,
            "combined_text": combined_text,
            "trending_keywords": keywords,
            "sources": [r["url"] for r in all_results if r.get("url")],
            "success": True,
        }

    except Exception as e:
        # Return a graceful fallback so the pipeline can continue
        return {
            "job_title": job_title,
            "raw_results": [],
            "combined_text": f"Research unavailable: {e}",
            "trending_keywords": [],
            "sources": [],
            "success": False,
            "error": str(e),
        }
