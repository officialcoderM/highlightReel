from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import warnings
warnings.filterwarnings("ignore")

print("Step 1: Loading documents...")
with open("documents.txt", "r") as f:
    documents = f.read()

print(f"Document loaded: {len(documents)} characters")

print("Step 2: Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_text(documents)
print(f"Created {len(chunks)} chunks")

if len(chunks) == 0:
    print("ERROR: No chunks created. Check documents.txt file.")
    exit()

# note below is where the model is prepared and the model is trained to covert text into semantic vector representations  
#load tool 

print("Step 3: Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
#note below is where vectorstore is created and embeddings are generated for each chunk and vecrot store holds that vector database which it creates from the chunks and embeddings
# use tool + store in vector database
print("Step 4: Storing chunks in vector database...")
vectorstore = Chroma.from_texts(
    chunks,
    embeddings,
    metadatas=[{"source": f"chunk_{i}"} for i in range(len(chunks))]
    #len chunk = 3 range = 0,1,2 "source": f"chunk_{i}" creates lib for each chunk i = 0 → {"source": "chunk_0"}
)
print(f"Stored {len(chunks)} chunks in database")

print("Step 5: Loading language model for generation...")
model_name = "microsoft/phi-2" #load model huggingface
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) #load tokens for that model
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)## Loads the pretrained causal language model (neural network weights and architecture) from Hugging Face
device = torch.device("cpu") 
model.to(device) # move model to CPU  like beofre 
#this function takes the prompt and used tokenizer to create into tensors 
def generate_answer(prompt):
    #prompt is stored in RAM and created on runtime using user question and context from retrieved chunks
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    #convert the prompt into tensors and make sure it doesnt exceeed 2048 tokens
    # inputs is a dictionary with keys like input_ids and attention_mask which are tensors that represent the prompt in a format the model can understand 
    inputs = {k: v.to(device) for k, v in inputs.items()}
    #looping through each key value pair in input dictionary
    #key value pair key is either tensor as input_ids or attention_mask padding and v is vector representation
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        #when this is false top_k and top_p are ignored 
        #wont pick the most likely token give some randomenss which should be false for general RAGS 
        # chat gpt says this is due to microphi2 sounding stiff if false 
        pad_token_id=tokenizer.eos_token_id,
        #eos token for padding each model is diffent 
        repetition_penalty=1.1
        #1 is free to repeat 1.5+ penalizes repetition and encourages more diverse output
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # response has question as well as answer and we only want answer so we extract that part
    # Extract only the answer part
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
        #split returns list of question - answere and neagative index gives us last elemelement 

    else:
        answer = response[len(prompt):].strip()
        #Take the response string and remove the first len(prompt) characters, leaving the generated answer.
    return answer
# above is defense to handle multipole formats of response and make sure we get the answer part only


print("\n" + "="*50)
print("RAG System Ready! Ask questions about ski resorts.")
print("="*55 + "\n")

while True:
    question = input("\nEnter your question (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    
    if not question.strip():
        print("Please enter a valid question.")
        continue
    
    print("\nRetrieving relevant information...")
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    
    if not relevant_chunks:
        print("No relevant information found.")
        continue
    
    print("Generating answer...")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
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
        for i, chunk in enumerate(relevant_chunks):
            preview = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content
            print(f"  Source {i+1}: {preview}")
        print("="*60)
    except Exception as e:
        print(f"Error generating answer: {e}")