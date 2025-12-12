
import chromadb
import uuid

def main():
    print("Connecting to ChromaDB at 192.168.0.8:6001...")
    # Connect to the ChromaDB server
    client = chromadb.HttpClient(host='192.168.0.8', port=6001)

    # Collection name
    collection_name = "rag_test"

    # Get or create the collection
    # We use get_or_create so we don't start fresh every time if we don't want to
    # But for a clear demo, maybe we want to delete it first if it exists?
    # Let's keep it simple: get_or_create
    print(f"Getting or creating collection '{collection_name}'...")
    collection = client.get_or_create_collection(name=collection_name)

    # Some sample data
    documents = [
        "The Moon is Earth's only natural satellite.",
        "The Moon interacts with the Earth causing tides.",
        "The first human landing on the Moon was in 1969 by Apollo 11.",
        "The distance from Earth to the Moon is about 384,400 km.",
        "The Moon has a very thin atmosphere called an exosphere."
    ]
    
    # Generate simple IDs
    ids = [f"id_{i}" for i in range(len(documents))]
    
    # Metadatas (optional)
    metadatas = [{"source": "manual_entry"} for _ in documents]

    print("Adding documents to collection...")
    # Add data to the collection
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    print(f"Successfully added {len(documents)} documents.")
    print("Current collection count:", collection.count())

if __name__ == "__main__":
    main()
