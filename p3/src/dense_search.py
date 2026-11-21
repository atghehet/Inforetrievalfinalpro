
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

texts = [
    "A neural network learns useful representations from data.",
    "A model is trained to recognize patterns in examples.",
    "The weather is sunny and warm today."
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(texts, normalize_embeddings=True)

query = "How does a machine learning model learn patterns?"
query_embedding = model.encode([query], normalize_embeddings=True)

scores = cosine_similarity(query_embedding, embeddings)[0]

ranked_indices = scores.argsort()[::-1]

print("\nQuery:", query)
print("\nRanked results:")

for rank, index in enumerate(ranked_indices, start=1):
    print(rank, round(float(scores[index]), 4), texts[index])
