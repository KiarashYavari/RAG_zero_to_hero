import config
import tiktoken
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# --------- indexing---------
# for more information about langchain integrations visit:
https://docs.langchain.com/oss/python/integrations/providers/overview

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

from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# or
# retriever = vectorstore.as_retriever()

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
