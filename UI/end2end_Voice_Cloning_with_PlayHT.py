import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

## API
import configparser
import os
from pinecone import Pinecone,ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain import LLMChain, PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from groq import Groq
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify, send_file, render_template
from pydub import AudioSegment
from pyht import Client, Format
from pyht.client import TTSOptions
import io

from flask import Flask, render_template, request, url_for
import time


config = configparser.ConfigParser()
config.read('config.ini')

# Get the API details
api_config = config['api']
pinecone_api_key = api_config['pinecone_api_key']
groq_api_key = api_config['groq_api_key']

# Required index
index_name = "knowledge-base"
llm_model = 'llama3-70b-8192'
stt_model = 'whisper-large-v3'

pc = Pinecone(api_key = pinecone_api_key)
index = pc.Index(index_name)

# Use LLaMA 3 for response generation
llm = ChatGroq(groq_api_key = groq_api_key, model_name=llm_model, temperature= 0.2)

# Load the model for embeddings
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

# Load the model for sparse matrix encoding
bm25_encoder = BM25Encoder().default()

# Using Hybrid Search Reriever for context extraction
retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index, top_k=5)

# Function to retrieve context using Pinecone
def context_retriever(user_query):

    context = retriever.invoke(user_query)

    return context



# Function to process audio query
def process_audio(api_key, audio_file_path, model_name):
    # Set the API key in the environment variable (temporarily)
    os.environ['GROQ_API_KEY'] = api_key

    # Initialize the Groq client
    client = Groq()

    # Open the audio file and send it for transcription
    with open(audio_file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_file_path, file.read()),
            model=model_name,
        )

    # Return the transcribed text
    return transcription.text

# Function to generate LLM response
def generate_llm_response(context, question, api_key, model_name):

    # Define the prompt template using PromptTemplate
    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant of Growexx AI Solutions. Your job is to generate answers for the asked question on the basis of context provided.
        The generated response will further be provided to a voice cloning text-to-speech model to convert the text into your company's
        CEO Vikas Agarwal's voice response. The voice accent is Indian english. If you have sufficient information available, try to generate
        response in atleast 30 to 40 words.

        Ensure the response is clear, concise, and includes appropriate punctuation like
        commas and full stops for natural speech pauses.

        If the required information is not available in the context, respond with:
        "The information you requested is not available in the provided context."

        Context: {context}

        Question: {question}

        Provide the answer with proper punctuation for voice clarity:
        """,
        input_variables=["context", "question"]
    )
#  You need to generate responses in not more than 50 words.
    # Chain for RAG (Retrieval-Augmented Generation)
    rag_chain = LLMChain(
        llm=llm,
        prompt=prompt_template
    )

    if not context:
        # Handle case where context is empty
        formatted_context = "No context available."
    else:
        formatted_context = "\n".join(str(item) for item in context)

    response = rag_chain.run(context=formatted_context, question=question)
    return response

# Function to generate audio response for each chunk and then concate into single longer response
def generate_wav_files(text):
  client = Client(
    user_id = 'pe0vomeSkkY10fVHmGb92mi9qYT2',
    api_key = '5dc0262d5ffe4fbeb12cc22914d1db4a',
  )

  options = TTSOptions(voice="s3://voice-cloning-zero-shot/b5b5fe11-17a8-4318-b353-3dee5d4829f7/original/manifest.json", sample_rate = 16000, format = Format.FORMAT_WAV, speed = 0.85, quality='QUALITY_PREMIUM')

  # Initialize an empty AudioSegment
  audio = AudioSegment.empty()

  # Append each chunk to the AudioSegment
  for chunk in client.tts(text=text, voice_engine="PlayHT2.0", options=options):

      segment = AudioSegment.from_raw(io.BytesIO(chunk), sample_width=2, frame_rate=16000, channels=1)

      audio += segment

  return audio


def produce_output(user_query):

    # Check if user_query is audio or text
    if user_query.endswith('.mp3'):
        text_query = process_audio(groq_api_key, user_query, stt_model)
    else:
        text_query = user_query

    # Retrieve context
    content = context_retriever(text_query)
    context = []
    for i in content:
        context.append(i.page_content)

    # Generate response using LLM
    text_response = generate_llm_response(context, text_query, groq_api_key, llm_model)

    audio_response_path = 'static/output.wav'

    play_ht_start_time = time.time()
    audio_response = generate_wav_files(text_response)
    play_ht_end_time = time.time()

    play_ht_time = round(play_ht_end_time - play_ht_start_time, 2)

    audio_response.export(audio_response_path, format = 'wav')

    return audio_response, text_response, play_ht_time


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    audio_response = None
    text_response = None
    text = None
    time_taken = None
    play_ht_time = None
    audio_response_path = 'output.wav'
    timestamp = int(time.time())  # Generate a unique timestamp


    if request.method == 'POST':
        text = request.form['user_query']
        print(f"text input: {text}")
        # Here you would generate the output.wav file
        start_time = time.time()
        audio_response, text_response, play_ht_time = produce_output(text)
        print(f"text response: {text_response}, length : {len(text_response)}")
        end_time = time.time()

        time_taken = round(end_time - start_time, 2)

    return render_template('index.html', user_query=text, llm_response=text_response, path=audio_response_path, timestamp=timestamp, time_taken = time_taken, play_ht_time = play_ht_time)

if __name__ == '__main__':
    app.run(debug=True)