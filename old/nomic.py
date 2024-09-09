import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
import requests

# Load image and text models
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)

# Image URLs
urls = {
    "Cat": 'https://cdn.pixabay.com/photo/2024/02/28/07/42/european-shorthair-8601492_640.jpg',
    "Dog": 'https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg',
    "Beach": 'https://images.pexels.com/photos/1032650/pexels-photo-1032650.jpeg',
    "Mountain": 'https://images.pexels.com/photos/1261728/pexels-photo-1261728.jpeg',
    "City": 'https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg'
}

# Function to get image embedding
def get_image_embedding(url):
    response = requests.get(url, stream=True)
    image = Image.open(response.raw)
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        img_emb = vision_model(**inputs).last_hidden_state
    return F.normalize(img_emb[:, 0], p=2, dim=1)

# Function to get text embedding
def get_text_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = text_model(**inputs)
    return F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)

# Generate embeddings for all images
image_embeddings = {name: get_image_embedding(url) for name, url in urls.items()}

# Function to rank similarities
def rank_similarities(query_embedding, embeddings):
    similarities = {name: F.cosine_similarity(query_embedding, emb).item() 
                    for name, emb in embeddings.items()}
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# Example 1: Text query
print("Example 1: Text query")
text_query = "A furry pet that likes to cuddle"
text_embedding = get_text_embedding(text_query)
rankings = rank_similarities(text_embedding, image_embeddings)

print(f"Query: '{text_query}'")
for name, similarity in rankings:
    print(f"{name}: {similarity:.4f}")

print("\n" + "="*50 + "\n")

# Example 2: Image query
print("Example 2: Image query")
image_query = "Beach"
rankings = rank_similarities(image_embeddings[image_query], image_embeddings)

print(f"Query: Image of {image_query}")
for name, similarity in rankings:
    print(f"{name}: {similarity:.4f}")

print("\n" + "="*50 + "\n")

# Example 3: Text query with mixed results
print("Example 3: Text query with mixed results")
text_query = "A place for a relaxing vacation"
text_embedding = get_text_embedding(text_query)
rankings = rank_similarities(text_embedding, image_embeddings)

print(f"Query: '{text_query}'")
for name, similarity in rankings:
    print(f"{name}: {similarity:.4f}")
