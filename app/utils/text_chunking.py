from __future__ import annotations


def chunk_text_by_words(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start_idx in range(0, len(words), step):
        chunk_words = words[start_idx : start_idx + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
    return chunks
