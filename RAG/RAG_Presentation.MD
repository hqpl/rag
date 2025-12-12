# Implementacja RAG z Lokalnym LLM (LM Studio) i ChromaDB

Ta prezentacja pokazuje krok po kroku, jak stworzyć prosty system RAG (Retrieval-Augmented Generation) wykorzystując lokalnie uruchomiony model (LM Studio) oraz bazę wektorową ChromaDB.

## 1. Wymagania wstępne

### Infrastruktura
- **ChromaDB**: Uruchomiona instancja dostępna pod adresem `192.168.0.8:6001`.
- **LM Studio**: Uruchomione lokalnie z załadowanym modelem (serwer API zgodny z OpenAI włączony na porcie `1234`). Serwer API musi zostać włączony ręcznie w zakładce Developer.

### Oprogramowanie
- Python
- `uv` (menedżer pakietów)

## 2. Inicjalizacja Projektu

Zarządzamy bibliotekami przy użyciu `uv`.

```bash
# Inicjalizacja projektu
uv init .

# Instalacja wymaganych bibliotek
uv add chromadb openai
```

## 3. Zapisywanie danych do Bazy (Krok 1)

Pierwszym krokiem jest "nakarmienie" naszej bazy wiedzy. Używamy skryptu `seed_db.py`.

Tworzy on kolekcję `rag_test` i dodaje do niej kilka faktów o Księżycu.

**Kod: `seed_db.py`**
```python
import chromadb

client = chromadb.HttpClient(host='192.168.0.8', port=6001)
collection = client.get_or_create_collection(name="rag_test")

documents = [
    "The Moon is Earth's only natural satellite.",
    "The Moon interacts with the Earth causing tides.",
    "The first human landing on the Moon was in 1969 by Apollo 11.",
    "The distance from Earth to the Moon is about 384,400 km.",
    "The Moon has a very thin atmosphere called an exosphere."
]

# Dodajemy dokumenty (ID są generowane automatycznie lub ręcznie)
ids = [str(i) for i in range(len(documents))]
collection.add(documents=documents, ids=ids)

print("Dane zostały zapisane.")
```

**Uruchomienie:**
```bash
uv run seed_db.py
```

## 4. Odczyt i Generowanie Odpowiedzi (Krok 2)

Teraz połączymy wyszukiwanie z generowaniem tekstu. Skrypt `query_llm.py`:
1. Pobiera pytanie.
2. Wyszukuje najbardziej pasujące fragmenty w ChromaDB.
3. Buduje prompt zawierający kontekst.
4. Wysyła zapytanie do LM Studio.

**Kod: `query_llm.py`**
```python
import chromadb
from openai import OpenAI

# 1. Połączenie z bazą
chroma_client = chromadb.HttpClient(host='192.168.0.8', port=6001)
collection = chroma_client.get_collection(name="rag_test")

# 2. Pytanie
query = "How far is the moon?"
results = collection.query(query_texts=[query], n_results=2)
context = "\n".join(results['documents'][0])

# 3. Połączenie z LLM (LM Studio)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# 4. Zapytanie do modelu
response = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "system", "content": "Answer based on the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(f"Odpowiedź: {response.choices[0].message.content}")
```

**Uruchomienie:**
```bash
uv run query_llm.py
```

## 5. Podsumowanie

Dzięki temu podejściu:
- Twoje dane pozostają prywatne (na lokalnej ChromaDB).
- Twój model działa lokalnie (LM Studio).
- Kod jest prosty i łatwy w utrzymaniu dzięki `uv`.
