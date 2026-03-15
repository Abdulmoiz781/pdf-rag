import pickle

with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

print(f"Total chunks: {len(chunks)}")
print()

for i, chunk in enumerate(chunks[:3]):
    print(f"--- Chunk {i+1} (Page {chunk['page']}) ---")
    print(chunk['text'])
    print()