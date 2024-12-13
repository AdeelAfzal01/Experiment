import time
import openai
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_key")
openai.api_key = OPENAI_API_KEY

# Function to search the web using DuckDuckGo
def search_web(question, num_results=4):
    search = DuckDuckGoSearchResults(output_format="list")
    results = search.invoke(question)
    return results[:num_results]

# Function to format the document content
def format_document(documents):
    docs_list = []
    for doc in documents:
        try:
            docs_list.append(f"{doc['snippet']}")
        except:
            pass
    return docs_list

# Function to rank documents based on the question and the model
def rank_documents(question, documents, model_name='thenlper/gte-small'):
    model = SentenceTransformer(model_name)
    question_embedding = model.encode(question)
    doc_embeddings = model.encode(documents)

    scores = [util.cos_sim(question_embedding, doc_embedding).item() for doc_embedding in doc_embeddings]
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return ranked_docs

# Function to compile context from the ranked documents
def compile_context(ranked_docs, top_n=2):
    context = ""
    if isinstance(ranked_docs, list):
        context = "\n".join([doc for doc, score in ranked_docs[:top_n]])
    return context

# Function to format the prompt used by the LLM
def format_prompt():
    prompt = PromptTemplate.from_template("""
        Instructions : You are an intelligent bot. Answer the question based on the provided context. If the context does not include enough information, respond with "Insufficient information."
        Question : {question}
        Context : {context}
        """)
    return prompt

# Main RAG pipeline function
def RagPipeline(question, top_n=2, sleep_time=15):
    if question:
        try:
            # Searching Web
            web_docs = search_web(question)
            time.sleep(sleep_time)  # Rate limit

            if not web_docs:
                return {'Question': question, 'Error': 'No results retrieved from the web.'}

            # Formating documents by extracting the snippets
            formatted_docs = format_document(web_docs)

            # Ranking the documents
            ranked_docs = rank_documents(question, formatted_docs)

            # Select Top-N documents and compile context
            context = compile_context(ranked_docs, top_n)

            # Create Prompt
            prompt = format_prompt()

            # LLM initialization
            llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=1500, timeout=30, streaming=True)

            # Chain initialization
            chain = prompt | llm

            # Chain invoking
            response = chain.invoke({"question": question, "context": context})

            return {
                'Question': question,
                'Web Search': web_docs,
                'Ranked Docs': [{'Document': doc, 'Score': score} for doc, score in ranked_docs],
                'Prompt': prompt.input_variables,
                'Answer': response.content
            }
        except Exception as e:
            return {'Question': question, 'Error': str(e)}

# Main entry point for the script
if __name__ == "__main__":
    question = "Explain the impact of climate change on agriculture."
    response = RagPipeline(question)
    print(response)
