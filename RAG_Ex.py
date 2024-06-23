import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the pre-trained generative model (GPT-2)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Step 2: Prepare the documents
documents = [
    "Solar energy is renewable and abundant.",
    "Solar panels can reduce electricity bills.",
    "Solar energy systems require low maintenance.",
    "The initial cost of solar panels can be high.",
    "Solar energy helps reduce carbon footprint."
]

# Encode documents to vectors
encoded_docs = [tokenizer.encode(doc, add_special_tokens=False, return_tensors='pt') for doc in documents]

# Ensure consistent size for document embeddings
doc_embeddings = []
for doc in encoded_docs:
    with torch.no_grad():
        doc_tensor = model.transformer.wte(doc).mean(dim=1).squeeze().numpy()
        doc_embeddings.append(doc_tensor)

doc_embeddings = np.stack(doc_embeddings)

def retrieve_documents(query, top_k=2):
    query_tensor = tokenizer.encode(query, add_special_tokens=False, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model.transformer.wte(query_tensor).mean(dim=1).squeeze().numpy()
    similarities = cosine_similarity([query_embedding], doc_embeddings)
    indices = similarities.argsort()[0][-top_k:][::-1]
    return [documents[i] for i in indices]

def generate_response(query, retrieved_docs):
    augmented_input = query + " " + " ".join(retrieved_docs)
    inputs = tokenizer.encode(augmented_input, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example query
query = "What are the benefits of using solar energy?"

# Step 3: Retrieve relevant documents
retrieved_docs = retrieve_documents(query)
print("Retrieved Documents:")
for doc in retrieved_docs:
    print("-", doc)

# Step 4: Generate response using the augmented input
response = generate_response(query, retrieved_docs)
print("\nGenerated Response:")
print(response)
