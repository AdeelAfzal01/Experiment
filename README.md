# RAG Pipeline with Web Search and OpenAI Integration

This project demonstrates how to create a Retrieval-Augmented Generation (RAG) pipeline using web search, document ranking, and OpenAI's GPT-4 model to generate answers to questions based on external web content. The pipeline retrieves relevant documents, ranks them, and uses them as context for answering the question.

## Overview

The goal of this project is to combine the capabilities of:
- **Web Search**: Use DuckDuckGo's search API to retrieve web results based on a given question.
- **Document Ranking**: Rank the retrieved documents using Sentence-BERT to match the question with the most relevant content.
- **Question Answering**: Use OpenAI's GPT-4 model (via the Langchain API) to generate a coherent answer based on the ranked documents.

## Requirements

- **Python 3.x**
- **External Libraries**: The project uses the following libraries:
  - `openai` for interacting with the OpenAI GPT-4 model.
  - `langchain` for chaining together components like the prompt template and LLM.
  - `sentence-transformers` for document ranking.
  - `duckduckgo-search` for retrieving search results.
  - `dotenv` to securely load the OpenAI API key from an `.env` file.

You can install the required libraries using:
bash

pip install openai langchain sentence-transformers duckduckgo-search transformers python-dotenv

## Setup
Create a .env File: Create a .env file in the root directory of the project. In this file, store your OpenAI API key like this:

text
Copy code
OPENAI_API_key=your_openai_api_key
### Run the Script:
To run the script, execute the following command in your terminal:


python main.py
### Question Customization:
The script contains a predefined question: "Explain the impact of climate change on agriculture.". You can modify the question variable in the main.py script to ask different questions.

## How It Works

### Search the Web:
The function search_web(question, num_results) uses DuckDuckGo’s search API to retrieve search results related to the question. It fetches a maximum of num_results (default is 4).

### Format Documents:
The format_document(documents) function formats the web search results by extracting and preparing the snippets for further use.

### Rank Documents:
The rank_documents(question, documents) function uses the SentenceTransformer model to encode both the question and the retrieved documents into embeddings. It then computes cosine similarity between the question's embedding and each document's embedding. The documents are ranked based on this similarity.

###  Compile Context:
The compile_context(ranked_docs, top_n) function selects the top N ranked documents (default 2) and compiles them into a context string to be used by the language model.

### Generate Answer:
- The format_prompt() function creates a prompt template to instruct the language model to answer the question based on the given context.
- The RagPipeline(question) function orchestrates the entire pipeline: it performs web search, ranks documents, compiles context, generates a prompt, and invokes the language model to generate the answer.

### Handle Errors:
If the pipeline encounters any issues (e.g., no results retrieved, API errors), it catches exceptions and returns an error message.

## Code Structure
main.py:
Imports: The necessary libraries are imported at the top of the file.
## Functions

The code is modularized into the following functions:

- **`search_web(question, num_results)`**: Retrieves search results from DuckDuckGo based on the given question. The `num_results` parameter controls how many results are fetched (default is 4).
  
- **`format_document(documents)`**: Extracts and formats the relevant snippets from the web search results for further use.
  
- **`rank_documents(question, documents)`**: Uses the `SentenceTransformer` model to encode both the question and retrieved documents into embeddings. It computes the cosine similarity between the question's embedding and each document's embedding, ranking the documents based on similarity.
  
- **`compile_context(ranked_docs, top_n)`**: Selects the top N ranked documents (default is 2) and compiles them into a context string to be used by the language model.
  
- **`format_prompt()`**: Creates a prompt template to instruct the language model to answer the question based on the provided context.
  
- **`RagPipeline(question, top_n=2, sleep_time=15)`**: Orchestrates the entire Retrieval-Augmented Generation (RAG) pipeline, which includes:
  1. Web search using DuckDuckGo.
  2. Document ranking based on cosine similarity.
  3. Context creation from top-ranked documents.
  4. Prompt generation for the LLM (Large Language Model).
  5. Answer generation by the language model.

## Main Block

The entry point of the script is handled under the `if __name__ == "__main__":` block. This block:

1. Takes a predefined question (or any customized question) and passes it into the `RagPipeline`.
2. Prints the result after processing, which includes:
   - The question.
   - The web search results retrieved.
   - The ranked documents along with their similarity scores.
   - The generated prompt variables.
   - The final answer generated by GPT-4 based on the context.

## Example Output

After running the script, the output will include:

1. The **question** you asked.
2. The **web search results** retrieved.
3. The **ranked documents** and their similarity scores.
4. The **prompt variables** used by the language model.
5. The **answer** generated by GPT-4 based on the provided context.

### Example output:

```json
{
  "Question": "Explain the impact of climate change on agriculture.",
  "Web Search": [
    "Link 1: Summary of article",
    "Link 2: Summary of article"
  ],
  "Ranked Docs": [
    {"Document": "Document text here", "Score": 0.85},
    {"Document": "Another relevant document", "Score": 0.78}
  ],
  "Prompt": {
    "question": "Explain the impact of climate change on agriculture.",
    "context": "Climate change affects agriculture by causing extreme weather conditions like drought, flooding, etc."
  },
  "Answer": "Climate change affects agriculture by leading to more frequent and severe weather events, altering growing seasons, and reducing crop yields."
}
```

## Issues

### 1. Slow Response with `unsloth/Llama-3.2-1B-Instruct`
- **Problem**: The model was taking too much time to respond.
- **Solution**: Consider using a smaller model or improving the hardware.

### 2. Bad Request with HuggingFace Inference Endpoint
- **Problem**: The model requires a Pro subscription and token inclusion.
- **Solution**: Visit [hf.co/pricing](https://hf.co/pricing) for subscription details and ensure your HuggingFace token is included in the request.

### 3. Poor Response from `google/mt5-small`
- **Problem**: Response wasn’t satisfactory when using the `google/mt5-small` model.
- **Example**: 
  - **Question**: Why did China enact the one-child policy?
  - **Response**: `<extra_id_0>`
- **Solution**: Try using another model or check response handling settings.

### 4. Model Too Large for Auto-Loading (`google/flan-t5-xl`)
- **Problem**: The model is too large (11GB > 10GB) to load automatically.
- **Solution**: Use a smaller model or manually load the model in an appropriate environment.

### 5. Short Responses from `google/flan-t5-small`
- **Problem**: The response was limited to only a few words.
- **Solution**: Adjust settings for longer responses or check token limits.

### 6. Limited Content from DuckDuckGoSearchResults
- **Problem**: The content extracted by DuckDuckGoSearchResults is limited.
- **Solution**: Increase the number of search results or switch to another search API.