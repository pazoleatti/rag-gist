import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    {context}
    Question: {question}
    
    Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval, check without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step 2: Format documents into context string
    context = format_docs(docs)
    # Step 3: Format prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)
    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)
    # Step 5: Return the context
    return response.content

def create_retrieval_chain_with_lcel():
    """Create a retrieval chain using LCEL (Langchain expression language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non non-LCEL approach:
     - Declarative and composable: Easy to chain operations with pipe operator (|)
     - Built-in streaming: chain.stream() works out of the box
     - Built-in async: chain.ainvoke() and chain.astream() available
     - Batch processing: chain.batch() for multiple inputs
     - Type safety: Better integration with langchain's type system
     - Less code: More concise and readable
     - Reusable: Chain can be saved, shared, and composed with other chains
     - Better debugging: LangChain provides better observability tools
     """
    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    print("Retrieving...")

    query = "what is Pinecone in machine learning?"
    # =========================================================
    # Option 1: Use implementation without LCEL
    # =========================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)

    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # =========================================================
    # Option 2: Use implementation with LCEL (Better Approach)
    # =========================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL")
    print("=" * 70)
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)