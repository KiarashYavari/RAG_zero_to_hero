import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1} # k nearest neighbors
)

docs = retriever.invoke("What is Task Decomposition?")

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

# Chain
chain = prompt | llm


# Run
response = chain.invoke({
    "context": docs,
    "question": "What is Task Decomposition?"
})

print(response.content)

# Input Dictionary
#        │
#        ▼
# ChatPromptTemplate
#        │
#        ▼
# Formatted Prompt
#        │
#        ▼
# ChatGoogleGenerativeAI
#        │
#        ▼
# AIMessage
#        │
#        ▼
# StrOutputParser
#        │
#        ▼
# Python String (Final Answer)

