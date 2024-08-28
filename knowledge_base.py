import requests
import configparser
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings


# Function to extract all URLs from a page
def extract_urls(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = {a.get('href') for a in soup.find_all('a', href=True)}
    full_urls = [urljoin(base_url, link) for link in links]  # Ensure full URLs
    return full_urls

# Base URL
base_url = 'https://www.growexx.com'
all_urls = extract_urls(base_url)
print(f"Found {len(all_urls)} URLs.")

def filter_urls(urls, base_domain):
    valid_urls = {url for url in urls if urlparse(url).netloc == base_domain}  # Use set comprehension for uniqueness
    return valid_urls

# Filter URLs to keep only those from the same domain
base_domain = urlparse(base_url).netloc
filtered_urls = filter_urls(all_urls, base_domain)
print(f"Filtered down to {len(filtered_urls)} valid unique URLs.")

combined_text = ''

for url in filtered_urls:
    try:
        # Send a GET request to the website
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the text from the HTML
            text = soup.get_text(separator=' ', strip=True)
            
            # Append the extracted text to the combined_text string
            combined_text += text + '\n'

            print("Extracted Data from URL: ", url)
            
        else:
            print(f"Failed to retrieve {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")

# Write the combined text to knowledge_base.txt
with open('knowledge_base.txt', 'w', encoding='utf-8') as file:
    file.write(combined_text)

print("Text extraction and merging complete. Data saved to knowledge_base.txt")

bm25_encoder =BM25Encoder().default()
bm25_encoder

with open('/content/knowledge_base.txt', 'r') as file:
    text_data = file.read()  # Read the entire content as a single string

def split_into_chunks(text, chunk_size=300, overlap=100):
    words = text.split()  # Split text into words
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = split_into_chunks(text_data, chunk_size=300, overlap=100)  # Adjust chunk_size and overlap as needed

bm25_encoder.fit(chunks)

bm25_encoder.dump("bm25_values.json")

bm25_encoder= BM25Encoder().load("bm25_values.json")

embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
embeddings

# Step 4: Initialize and connect to Pinecone
# Read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the API details
api_config = config['api']
pinecone_api_key = api_config['pinecone_api_key']

# Initialize pinecone client
pc = Pinecone(api_key=pinecone_api_key)

index_name = 'knowledge-base'

if index_name not in pc.list_indexes():
    pc.create_index(index_name, 
                    dimension=768,
                    metric='dotproduct',
                    spec = ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    ))

index = pc.Index(index_name)

retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index, top_k=10)

retriever.add_texts(chunks)
