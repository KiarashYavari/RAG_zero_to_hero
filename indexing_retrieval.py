import config
import tiktoken
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# --------- indexing---------
# for more information about langchain integrations visit:
# https://docs.langchain.com/oss/python/integrations/providers/overview

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

# Split

# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough.
# The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible,
# as those would generically seem to be the strongest semantically related pieces of text.


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)


embd = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embd
)

# retriever = vectorstore.as_retriever()
# it uses these default settings:
# - search_type="similarity"
# - k=4 (retrieves up to four relevant documents)

# or

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1} # k nearest neighbors
)

# splits (document chunks)
#           │
#           ▼
# Gemini embed_documents()
#           │
#           ▼
# Embedding vectors
#           │
#           ▼
# Chroma Vector Database

# then you perform a search:
docs = retriever.invoke("What is Task Decomposition?")

# documents content: (it returns a list but in this case I only have one)
print(docs[0].page_content)
# document metadata
print(docs[0].metadata)

# output >>>
# Component One: Planning#
# A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.
# Task Decomposition#
# Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks.
# The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. 
# CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.
# Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. 
# It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. 
# The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.
# Task decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\n1.", 
# "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. 
# "Write a story outline." for writing a novel, or (3) with human inputs.
# {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}

# what happens behind the scenes?
# retriever.invoke(question)
#            │
#            ▼
# Gemini generates question embedding
#            │
#            ▼
# Chroma performs vector similarity search
#            │
#            ▼
# Returns the top k relevant documents

# ** Since I previously configured our retriever with k=1, 
# it will return at most one document chunk.

# | Method                                                    | Purpose |
# |---                                                          |---|
# | `retriever.invoke(question)`                              | Synchronous retrieval |
# | `await retriever.ainvoke(question)`                       | Asynchronous retrieval |
# | `retriever.batch([question1, question2])`                 | Retrieve documents for multiple questions |
# | `vectorstore.similarity_search(question, k=3)`            | Search the vector database directly |
# | `vectorstore.similarity_search_with_score(question, k=3)` | Retrieve documents along with distance scores |

# ** At this stage, you are only performing the Retrieval (R) part of RAG. 
# We haven't passed the retrieved documents to Gemini's generative model to produce an answer yet.
