
import chromadb
from openai import OpenAI

def main():
    # 1. Connect to ChromaDB
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.HttpClient(host='192.168.0.8', port=6001)
    collection = chroma_client.get_collection(name="rag_test")

    # 2. Define the Query
    query_text = "How far is the moon?"
    print(f"\nQuestion: {query_text}")

    # 3. Retrieve relevant documents from ChromaDB
    print("Searching for relevant documents...")
    items = collection.query(
        query_texts=[query_text],
        n_results=2  # Get top 2 results
    )
    
    # Flatten the results (list of lists)
    documents = items['documents'][0]
    
    print("\nFound context:")
    for doc in documents:
        print(f"- {doc}")

    # 4. Prepare the Prompt for LM Studio
    context_str = "\n".join(documents)
    
    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question."
    user_prompt = f"""
    Context:
    {context_str}
    
    Question: 
    {query_text}
    """

    # 5. Connect to LM Studio (via OpenAI client)
    # LM Studio usually runs on port 1234
    print("\nConnecting to LM Studio (local LLM)...")
    client = OpenAI(
        base_url="http://localhost:1234/v1", 
        api_key="lm-studio"  # Any string works for LM Studio
    )

    # 6. Get Response
    print("Asking LLM...")
    response = client.chat.completions.create(
        model="model-identifier", # This is often ignored by LM Studio if only one model is loaded, or use the loaded model ID
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    print("\n--- LLM Answer ---")
    print(answer)
    print("------------------")

if __name__ == "__main__":
    main()
