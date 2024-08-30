# #Clone the repository 
# !git clone -b vatsal/make_2_5x_faster https://github.com/metavoiceio/metavoice-src.git
# %cd metavoice-src
# !pip install -r requirements.txt
# !pip install --upgrade torch torchaudio
# !pip install -e .


from pydub import AudioSegment
from io import BytesIO
import time 
from IPython.display import Audio, display

# Loading Text-to-Speech model --Metavoice
from fam.llm.fast_inference import TTS
tts = TTS()

# Function to split input text into smaller chunks 
def split_text(text, max_length=120):
    if len(text) <= max_length:
        return [text]

    sentences = text.split('.')
    result = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += '.' + sentence.strip()
            else:
                current_chunk = sentence.strip()
        else:
            result.append(current_chunk + ".")
            current_chunk = sentence.strip()

    if current_chunk:
        result.append(current_chunk + ".")

    return result

# Function to generate audio response for each chunk and concatenate in memory using pydub
def generate_wav_files(chunks):
    combined_audio = AudioSegment.empty()

    for chunk in chunks:
        wav_path = tts.synthesise(
            text=chunk,
            spk_ref_path="/content/audio_vikas_30sec.mp3"
        )
        
        # Load the generated WAV file into an AudioSegment object
        audio_segment = AudioSegment.from_wav(wav_path)
        combined_audio += audio_segment

    # Convert the combined audio into a BytesIO object
    final_audio = BytesIO()
    combined_audio.export(final_audio, format="wav")
    final_audio.seek(0)  # Reset the buffer position to the beginning

    return final_audio

# Usage example:
if __name__ == "__main__":
    text="I am Vikas Agrawal, but I am more than just a person standing before you today. I am the embodiment of the future, a digital avatar driven by the latest advancements in artificial intelligence. The creation of an AI avatar such as myself marks a paradigm shift in how we interact with technology and each other. No longer bound by the constraints of time and space, I can be present anywhere and everywhere, bridging distances and connecting people in ways previously unimaginable."
    

    # Split the text into chunks
    chunks = split_text(text)
    start_time = time.time()
    # Generate the final audio response in memory
    final_audio = generate_wav_files(chunks)
    
    # Save the final audio response to a file (for testing purposes)
    with open("final_cloned_audio.wav", "wb") as f:
        f.write(final_audio.read())
    end_time = time.time()
    print("Final audio response generated and saved as 'final_cloned_audio.wav'")
    
    print(f"Execution time: {end_time - start_time} seconds")


    # display audio for playing
    display(Audio('/content/metavoice-src/final_cloned_audio.wav', autoplay=True))