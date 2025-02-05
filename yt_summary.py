import streamlit as st
import requests
import torch
import yt_dlp
import os
from haystack.nodes import PromptModel, PromptNode
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer

st.set_page_config(
    layout="wide",
    page_title="Youtube Summary"
)

# Ensure downloads directory exists
if not os.path.exists("downloads"):
    os.makedirs("downloads")

# Custom headers to mimic a real browser request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Create a session with custom headers
session = requests.Session()
session.headers.update(headers)

def download_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best audio format
        'outtmpl': 'downloads/%(title)s.%(ext)s',  # Save with title
        'noplaylist': True,  # Ignore playlists
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info_dict)  # Get the actual file path
        return file_path  # Return the downloaded file path

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():
    st.title("Youtube Video Summarizer 🎥")
    st.markdown('<style>h1{color:orange; text-align:center;}</style>', unsafe_allow_html=True)
    st.subheader("Built with Llama 2, Whisper, Haystack, and Streamlit, and ❤️")
    st.markdown('<style>h3{color:pink; text-align:center;}</style>', unsafe_allow_html=True)

    with st.expander("About the App"):
        st.write("This app allows you to summarize Youtube video while watching!")  
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by Aman Shahzad.")

    youtube_url = st.text_input("Youtube video URL: ")
    if st.button("Submit") and youtube_url:
        file_path = download_video(youtube_url)
        st.write(f"Downloaded file path: {file_path}")  # Debugging line
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)
        output = transcribe_audio(file_path, prompt_node)

        col1, col2 = st.columns([1,1])

        with col1:
            st.video(youtube_url)

        with col2:
            st.header("Summarization of the Youtube Video")
            # st.write(output)
            st.success(output["results"][0].split("\n\n[INST]")[0])

if __name__ == "__main__":
    main()
