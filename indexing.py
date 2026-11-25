import config
import tiktoken
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

# count the number of tokens >>> contenxt limits
# best amount of overlap

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(question, "cl100k_base")) # choose the best lib and method for tokenization 
# aligning with your goals and needs

# Initialize Gemini embeddings
embd = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    # output_dimensionality=768
)

# Generate embeddings
query_result = embd.embed_query(question)
# embed_documents() accepts a list of documents 
# and returns a list of embedding vectors, one for each document.
document_result = embd.embed_documents([document])[0]
# returns a list of vector store one for each document
# print(embd.embed_documents([document]))
# somthing like: [    [0.12, -0.35, 0.87, 0.45]  ] but much bigger

# Check embedding dimensions
print(len(query_result))
print(len(document_result))

# -----------cosine similarity--------------
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)