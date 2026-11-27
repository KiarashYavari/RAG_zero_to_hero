import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client


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

# clear out the metadata to put only the content in the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
#The temperature parameter controls how much randomness the LLM uses when selecting the next token during text generation.
# - Low temperature: More predictable, consistent, and focused responses.
# - High temperature: More diverse, creative, and unpredictable responses.
# The LLM generates a probability distribution for possible next tokens. 
# Temperature modifies that distribution.

# Temperature	        Behavior	            Suitable use cases
# 0.0	                Highly predictable	    RAG, information extraction
# 0.2	                Focused	                Question answering, code generation
# 0.5	                Balanced	            General conversation
# 0.7	                More diverse	        Content generation
# 1.0	                Unscaled logits	        General-purpose generation
# 1.5–2.0	            Highly random	        Creative experimentation, where supported

# Chain
chain = prompt | llm


# Run
response = chain.invoke({
    "context": format_docs(docs),
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

# we can dowload the prompt template from langchain hub:
client = Client()

financial_rag_prompt = client.pull_prompt("my-company/financial-rag")

print(prompt_hub_rag)

# LangChain Hub is a repository where developers can publish, share, discover, and reuse prompt templates. 
# Its prompt management functionality is now integrated into LangSmith.

# The main use cases are:
# - Reusing existing prompt templates.
# - Sharing prompts with other developers.
# - Managing and versioning prompts independently of application code.
# - Experimenting with different prompting techniques.
# - Maintaining centralized prompts across multiple applications.

# The returned object is a ChatPromptTemplate, 
# which is the same type of object we created manually earlier in our RAG project.

