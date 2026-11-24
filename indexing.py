import config
import tiktoken


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

