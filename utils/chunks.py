
def chunk_text(text: str, chunk_size: int =500, overlap: int = 50) -> list[str]:
    
    chunks = list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks