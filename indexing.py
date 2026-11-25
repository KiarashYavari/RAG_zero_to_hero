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

# -----------indexing--------------
# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # We are instructing BeautifulSoup to parse elements matching any
            # of these three HTML classes, along with their nested content.
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()
# The extracted text would approximately look like this:
# LLM Powered Autonomous Agents

# This article explains autonomous AI agents.
# AI agents use tools, memory, and planning.

# | Component           | Purpose |
# |---                  |---|
# | `WebBaseLoader`     | Downloads the webpage and converts its extracted text into LangChain documents. |
# | `bs4`               | BeautifulSoup library for parsing HTML. |
# | `SoupStrainer`      | Restricts which parts of the HTML are parsed. |
# | `parse_only`        | Passes the filtering criteria to BeautifulSoup. |
# | `class_`            | Filters HTML elements based on their CSS classes. |
# | `loader.load()`     | Returns a list of LangChain `Document` objects. |

# we need to remove unwated text and html tags from webpages!

# What if the website generates content using JavaScript?
# There is an important difference between removing JavaScript and executing JavaScript.
# Some websites built with React, Angular, or Vue may initially return HTML like this:
# <body>
#     <div id="root"></div>
#     <script src="app.js"></script>
# </body>

# The actual content is generated after JavaScript executes in the browser.
# BeautifulSoup does not execute JavaScript. Therefore, you might receive an almost empty document.
# For these websites, you can use Playwright to render the webpage in a headless browser, then pass the rendered HTML to BeautifulSoup for cleaning.

# Tool	            Purpose	Executes                    JavaScript
# BeautifulSoup	    HTML parsing and filtering	          No
# WebBaseLoader	    Simple webpage loading	              No
# Trafilatura	    Automatic main-content extraction	  No
# Playwright	    Browser automation and rendering	  Yes
# Selenium	        Browser automation and rendering	  Yes

# Webpage URL
#      │
#      ▼
# Fetch HTML (Requests / Playwright)
#      │
#      ▼
# Parse HTML (BeautifulSoup)
#      │
#      ▼
# Remove irrelevant elements
#      │
#      ▼
# Extract meaningful text
#      │
#      ▼
# Create LangChain Documents
#      │
#      ▼
# Text splitting (Chunking)
#      │
#      ▼
# Gemini Embeddings
#      │
#      ▼
# Vector Database

