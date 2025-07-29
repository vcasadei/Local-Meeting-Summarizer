import os
import argparse
import whisper
import requests
import json
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# --- Configuration ---
# URL for the Ollama API endpoint. It's usually running on localhost.
OLLAMA_URL = "http://localhost:11434/api/generate"
# Default model to use for summarization. Make sure you have pulled this model with "ollama run <model_name>"
DEFAULT_MODEL = "llama3"
# Supported audio and video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.mov', '.avi']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac']

def extract_audio(video_path: str) -> str:
    """
    Extracts the audio from a video file and saves it as a temporary MP3 file.

    :param video_path: Path to the video file.
    :return: The path to the extracted audio file.
    """
    print(f"Extracting audio from '{video_path}'...")
    try:
        video = VideoFileClip(video_path)
        audio_path = "temp_audio.mp3"
        video.audio.write_audiofile(audio_path, codec='mp3')
        video.close()
        print(f"Audio extracted successfully to '{audio_path}'.")
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path: str) -> (str, str):
    """
    Transcribes the given audio file using the Whisper model.

    :param audio_path: Path to the audio file.
    :return: A tuple containing the detected language and the full transcript.
    """
    print("Loading Whisper model...")
    # Using the "base" model is a good balance of speed and accuracy for an MVP.
    # For higher accuracy, you can use "medium" or "large", but they are slower.
    model = whisper.load_model("base")
    print("Model loaded. Starting transcription (this may take a while)...")
    
    # The 'fp16=False' option might be necessary for CPU-only execution.
    # If you have a compatible GPU, you can remove it for a speed boost.
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    
    print("Transcription complete.")
    return result["language"], result["text"]

def summarize_text_with_ollama(text: str, model: str) -> str:
    """
    Sends the transcript to Ollama to generate a summary.

    :param text: The full transcript of the meeting.
    :param model: The name of the Ollama model to use.
    :return: The generated summary as a string.
    """
    print(f"Sending transcript to Ollama using model '{model}' for summarization...")

    # This is the prompt that instructs the LLM on how to behave.
    # Good prompting is key to getting good results.
    prompt = f"""
    You are an expert assistant specialized in summarizing meetings.
    Your task is to create a concise and structured summary of the following meeting transcript.

    Please provide the summary in three sections:
    1.  **Key Topics Discussed**: A brief overview of the main points and subjects covered.
    2.  **Decisions Made**: A bulleted list of any decisions that were agreed upon.
    3.  **Action Items**: A bulleted list of tasks assigned to individuals, including who is responsible if mentioned.

    If any of these sections are not applicable (e.g., no decisions were made), state that clearly.
    Do not add any commentary or information that was not present in the transcript.

    Here is the transcript:
    ---
    {text}
    ---
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # We want the full response at once
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # The response from Ollama is a stream of JSON objects, one per line.
        # We parse the final one to get the complete response.
        response_lines = response.text.strip().split('\n')
        final_response = json.loads(response_lines[-1])
        
        print("Summary received from Ollama.")
        return final_response.get("response", "Error: Could not parse summary from Ollama response.")

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}\nIs the Ollama server running?"
    except json.JSONDecodeError:
        return "Error: Could not decode the JSON response from Ollama."


def main():
    """
    Main function to orchestrate the transcription and summarization process.
    """
    parser = argparse.ArgumentParser(description="Transcribe and summarize a meeting recording.")
    parser.add_argument("file_path", help="Path to the video or audio file of the meeting.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"The Ollama model to use for summarization (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    file_path = args.file_path
    ollama_model = args.model

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    audio_path = None
    is_temp_audio = False

    if file_ext in SUPPORTED_VIDEO_FORMATS:
        audio_path = extract_audio(file_path)
        if audio_path is None:
            return
        is_temp_audio = True
    elif file_ext in SUPPORTED_AUDIO_FORMATS:
        audio_path = file_path
    else:
        print(f"Error: Unsupported file format '{file_ext}'.")
        print(f"Supported video: {SUPPORTED_VIDEO_FORMATS}")
        print(f"Supported audio: {SUPPORTED_AUDIO_FORMATS}")
        return

    try:
        language, transcript = transcribe_audio(audio_path)
        print(f"\n--- Detected Language: {language.upper()} ---")
        print("\n--- Full Transcript ---")
        print(transcript)

        if transcript:
            summary = summarize_text_with_ollama(transcript, ollama_model)
            print("\n" + "="*50)
            print("          MEETING SUMMARY")
            print("="*50 + "\n")
            print(summary)
        else:
            print("\nTranscript is empty, cannot generate summary.")

    finally:
        # Clean up the temporary audio file if one was created
        if is_temp_audio and os.path.exists(audio_path):
            print(f"\nCleaning up temporary file '{audio_path}'...")
            os.remove(audio_path)

if __name__ == "__main__":
    main()
