from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_semantic_similarity(query1: str, query2: str) -> float:
    """Compute similarity between two queries."""
    embeddings = model.encode([query1, query2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity.item())

def max_similarity_against_history(current_query: str, past_queries: list[str]) -> float:
    """Return the maximum similarity between current_query and all past queries."""
    if not past_queries:
        return 0.0  

    query_pairs = [[current_query, q] for q in past_queries]
    all_queries = [current_query] + past_queries
    embeddings = model.encode(all_queries, convert_to_tensor=True)

    current_embedding = embeddings[0]
    past_embeddings = embeddings[1:]

    similarities = util.cos_sim(current_embedding, past_embeddings)
    max_score = float(similarities.max().item())
    return max_score
