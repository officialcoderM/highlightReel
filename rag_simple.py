import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

print("Step 1: Loading documents...")
with open("documents.txt", "r") as f:
    documents = f.read()

print(f"Document loaded: {len(documents)} characters")

print("Step 2: Splitting documents into chunks...")
# Simple and reliable chunking
chunk_size = 500
chunks = []
sentences = documents.replace('\n', ' ').split('. ')

current_chunk = ""
for sentence in sentences:
    # Add period back if not empty
    if sentence:
        sentence = sentence + ". "
    
    if len(current_chunk) + len(sentence) <= chunk_size:
        current_chunk += sentence
    else:
        if current_chunk:
            chunks.append(current_chunk.strip())
        current_chunk = sentence

# Add the last chunk
if current_chunk:
    chunks.append(current_chunk.strip())

print(f"Created {len(chunks)} chunks")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {chunk[:100]}...")

if len(chunks) == 0:
    print("ERROR: No chunks created. Check documents.txt file.")
    exit()

print("Step 3: Creating embeddings...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(chunks)
print(f"Created {len(embeddings)} embeddings")

print("Step 4: Storing in vector database...")
# Clear any existing ChromaDB data
client = chromadb.Client()

# Delete existing collection if it exists
try:
    client.delete_collection("ski_resorts")
    print("Deleted existing collection")
except:
    pass

# Create new collection
collection = client.create_collection("ski_resorts")

# Add chunks to collection
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding.tolist()],
        documents=[chunk],
        metadatas=[{"source": f"chunk_{i}", "index": i}]
    )

print(f"Stored {len(chunks)} chunks in database")

print("Step 5: Loading language model for generation...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cpu")
model.to(device)

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response[len(prompt):].strip()
    return answer

print("\n" + "="*60)
print("RAG System Ready! Ask questions about ski resorts.")
print("="*60 + "\n")

while True:
    question = input("\nEnter your question (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    if not question.strip():
        print("Please enter a valid question.")
        continue
    
    print("\nRetrieving relevant information...")
    question_embedding = embedding_model.encode([question])
    
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=3
    )
    
    if not results['documents'][0]:
        print("No relevant information found.")
        continue
    
    print("Generating answer...")
    context = "\n\n".join(results['documents'][0])
    
    prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have that information in my documents."

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        answer = generate_answer(prompt)
        
        print("\n" + "="*60)
        print(f"Question: {question}")
        print(f"\nAnswer: {answer}")
        print("\nSources:")
        for i, doc in enumerate(results['documents'][0]):
            preview = doc[:150] + "..." if len(doc) > 150 else doc
            print(f"  Source {i+1}: {preview}")
        print("="*60)
    except Exception as e:
        print(f"Error generating answer: {e}")