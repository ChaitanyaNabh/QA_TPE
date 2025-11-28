import re
from typing import List, Dict


def find_context_matches(pdfs: Dict[str, dict], context: str, top_k: int = 5) -> List[dict]:
    """Search PDF page texts for sentences that match a given context string.

    - `pdfs` should be the same structure used in `st.session_state.pdfs` from `New.py`.
    - Returns a list of dicts: {pdf_name, page, score, text} ordered by score.
    """
    if not context or not pdfs:
        return []

    # simple keyword extraction from context
    keywords = re.findall(r"\w+", context.lower())
    if not keywords:
        return []

    results = []

    for pdf_name, info in pdfs.items():
        page_texts = info.get("page_texts")

        # If page_texts are not precomputed, attempt extraction from reader
        if page_texts is None:
            reader = info.get("reader")
            if not reader:
                continue
            page_texts = []
            for i in range(info.get("pages", 0)):
                try:
                    text = reader.pages[i].extract_text() or ""
                except Exception:
                    text = ""
                page_texts.append(text)
            info["page_texts"] = page_texts

        for page_idx, text in enumerate(page_texts):
            if not text:
                continue

            # Split into sentences heuristically
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                low = sent.lower()
                # count how many keywords occur in the sentence
                count = sum(1 for kw in keywords if kw in low)
                if count > 0:
                    # score: occurrences normalized by sentence length (words)
                    words = len(sent.split())
                    score = count / (words + 1)
                    results.append({
                        "pdf_name": pdf_name,
                        "page": page_idx,
                        "score": score,
                        "text": sent.strip()
                    })

    # sort by score and truncate
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
