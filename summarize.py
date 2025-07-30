# Save as: summarize_fast.py
import os
import argparse
import requests
import json
from tqdm import tqdm

# --- Dynamic Imports ---
# We will import the transcription library based on user choice
whisper = None
faster_whisper = None

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac']

def check_ollama_status() -> bool:
    """Checks if the Ollama server is running and reachable."""
    print("Checking connection to Ollama server...")
    try:
        ollama_base_url = OLLAMA_URL.replace("/api/generate", "")
        response = requests.get(ollama_base_url, timeout=5)
        response.raise_for_status()
        print("Ollama server is reachable.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n--- Ollama Connection Error ---\nCould not connect to the Ollama server at '{ollama_base_url}'.\nPlease ensure the Ollama application is running.\nError details: {e}\n---------------------------------\n")
        return False

def transcribe_audio(audio_path: str, use_fast_whisper: bool, language: str) -> (str, str):
    """
    Transcribes the audio file using either the original or the fast Whisper implementation.

    :param audio_path: Path to the audio file.
    :param use_fast_whisper: Boolean flag to select the transcription engine.
    :param language: The language for transcription (e.g., 'en', 'pt'). None for auto-detect.
    :return: A tuple containing the detected language and the full transcript.
    """
    model_name = "base"
    transcribe_options = {"language": language} if language else {}
    
    if use_fast_whisper:
        # --- Using faster-whisper ---
        global faster_whisper
        if faster_whisper is None:
            from faster_whisper import WhisperModel
            faster_whisper = WhisperModel
            
        print(f"Loading faster-whisper model '{model_name}' (this may download the model on first run)...")
        model = faster_whisper(model_name, device="auto", compute_type="default")
        
        print(f"Model loaded. Starting transcription (faster-whisper) for language: {language or 'auto'}...")
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True, **transcribe_options)
        
        transcript_parts = []
        total_duration = round(info.duration, 2)
        with tqdm(total=total_duration, unit=" seconds", desc="Transcribing") as pbar:
            for segment in segments:
                transcript_parts.append(segment.text)
                pbar.update(segment.end - segment.start)
        
        transcript = "".join(transcript_parts).strip()
        detected_language = info.language
        
    else:
        # --- Using original whisper ---
        global whisper
        if whisper is None:
            import whisper as original_whisper
            whisper = original_whisper

        print(f"Loading original whisper model '{model_name}'...")
        model = whisper.load_model(model_name)
        print(f"Model loaded. Starting transcription for language: {language or 'auto'}...")
        result = model.transcribe(audio_path, fp16=False, verbose=False, **transcribe_options)
        transcript = result["text"]
        detected_language = result["language"]

    print("Transcription complete.")
    return detected_language, transcript

def get_prompt(language: str, text: str) -> str:
    """
    Generates the appropriate prompt for Ollama based on the selected language.

    :param language: The language for the summary ('pt' for Portuguese).
    :param text: The transcript to be summarized.
    :return: The fully formatted prompt string.
    """
    if language == 'pt':
        return f"""
        Você é um assistente especialista em resumir reuniões.
        Sua tarefa é criar um resumo conciso e estruturado da seguinte transcrição de reunião, em Português do Brasil.
        Não me faça perguntas, somente faça o resumo em Português do Brasil.

        Por favor, forneça o resumo em três seções:
        1.  **Principais Tópicos Discutidos**: Uma visão geral dos principais pontos e assuntos abordados.
        2.  **Decisões Tomadas**: Uma lista com marcadores de quaisquer decisões que foram acordadas.
        3.  **Itens de Ação**: Uma lista com marcadores de tarefas atribuídas a indivíduos, incluindo quem é o responsável, se mencionado.

        Se alguma destas seções não for aplicável (por exemplo, nenhuma decisão foi tomada), declare isso claramente.
        Não adicione nenhum comentário ou informação que não estivesse presente na transcrição.

        Aqui está a transcrição:
        ---
        {text}
        ---
        """
    else: # Default to English
        return f"""
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

def summarize_text_with_ollama(text: str, model: str, language: str) -> str:
    """Sends the transcript to Ollama to generate a summary in the specified language."""
    print(f"Sending transcript to Ollama using model '{model}' for summarization with language {language}...")
    
    prompt = get_prompt(language, text)

    if language == 'pt':
        model = "brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16"

    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        response_lines = response.text.strip().split('\n')
        final_response = json.loads(response_lines[-1])
        print("Summary received from Ollama.")
        return final_response.get("response", "Error: Could not parse summary from Ollama response.")
    except requests.exceptions.RequestException as e:
        return f"Error during summarization request to Ollama: {e}"
    except json.JSONDecodeError:
        return "Error: Could not decode the JSON response from Ollama."

def main():
    """Main function to orchestrate the process."""
    parser = argparse.ArgumentParser(description="Transcribe and summarize a meeting recording with performance options.")
    parser.add_argument("file_path", help="Path to the video or audio file of the meeting.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"The Ollama model to use for summarization (default: {DEFAULT_MODEL}).")
    parser.add_argument("--fast", action="store_true", help="Use the faster-whisper implementation for transcription.")
    parser.add_argument("--language", type=str, default=None, help="Language of the meeting audio (e.g., 'en', 'pt', 'es'). If not specified, Whisper will auto-detect.")
    args = parser.parse_args()

    if not check_ollama_status():
        return

    if args.fast:
        try:
            import faster_whisper
        except ImportError:
            print("\n--- Dependency Error ---")
            print("The '--fast' option requires the 'faster-whisper' library.")
            print("Please install it by running: pip install faster-whisper")
            print("------------------------\n")
            return

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    file_ext = os.path.splitext(file_path)[1].lower()
    audio_path, is_temp_audio = None, False

    if file_ext in SUPPORTED_AUDIO_FORMATS:
        audio_path = file_path
    else:
        print(f"Error: Unsupported file format '{file_ext}'.")
        return

    try:
        detected_language, transcript = transcribe_audio(audio_path, args.fast, args.language)
        print(f"\n--- Detected Language: {detected_language.upper()} ---")
        print("\n--- Full Transcript ---")
        print(transcript)

        if transcript:
            summary_lang = args.language if args.language else detected_language
            summary = summarize_text_with_ollama(transcript, args.model, summary_lang)
            print("\n" + "="*50 + "\n          MEETING SUMMARY\n" + "="*50 + "\n")
            print(summary)
        else:
            print("\nTranscript is empty, cannot generate summary.")

    finally:
        if is_temp_audio and os.path.exists(audio_path):
            print(f"\nCleaning up temporary file '{audio_path}'...")
            os.remove(audio_path)

if __name__ == "__main__":
    main()
