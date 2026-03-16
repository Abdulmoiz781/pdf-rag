from retriever import retrieve

query = "What optimizer was used for training?"

results = retrieve(query)

print(f"Query: {query}")
print(f"Found {len(results)} relevant chunks")
print()

for i, chunk in enumerate(results):
    print(f"--- Result {i+1} (Page {chunk['page']}) ---")
    print(chunk['text'])
    print()