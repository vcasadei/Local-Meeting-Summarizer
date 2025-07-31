# Save as: summarize_fast.py
import os
import argparse
import requests
import json
import sys
from tqdm import tqdm
from datetime import timedelta
import warnings

# --- Suppress Warnings ---
# Pyannote and its dependencies can generate a lot of non-critical warnings.
# This will make the output cleaner.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Dynamic Imports for optional features ---
whisper = None
faster_whisper = None
pyannote_audio = None
torch = None

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

def transcribe_and_diarize(
    audio_path: str, 
    language: str, 
    whisper_model_name: str, 
    hf_token: str = None,
    min_speakers: int = None,
    max_speakers: int = None
) -> str:
    """
    Performs speaker diarization and then transcribes, assigning speakers to text.

    :param audio_path: Path to the audio file.
    :param language: The language for transcription.
    :param whisper_model_name: The size of the Whisper model to use.
    :param hf_token: Optional Hugging Face Hub token.
    :param min_speakers: Minimum number of speakers.
    :param max_speakers: Maximum number of speakers.
    :return: A formatted string with speaker-labeled transcript.
    """
    global pyannote_audio, torch, faster_whisper
    if pyannote_audio is None or torch is None:
        import torch
        from pyannote.audio import Pipeline
        pyannote_audio = Pipeline
    
    if faster_whisper is None:
        from faster_whisper import WhisperModel
        faster_whisper = WhisperModel

    # 1. Diarization
    print("Step 1: Performing speaker diarization...")
    try:
        auth_token = hf_token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        diarization_pipeline = pyannote_audio.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=auth_token
        )
        
        if torch.cuda.is_available():
            diarization_pipeline = diarization_pipeline.to("cuda")
        
        print(f"Diarizing with settings: min_speakers={min_speakers}, max_speakers={max_speakers}")
        # The 'num_workers' parameter has been removed to fix the compatibility issue.
        diarization = diarization_pipeline(
            audio_path, 
            min_speakers=min_speakers, 
            max_speakers=max_speakers
        )
        print("Diarization complete.")
    except Exception as e:
        print(f"\n--- Pyannote Error ---\nFailed to run pyannote.audio pipeline: {e}\n------------------------\n")
        sys.exit(1)

    # 2. Transcription with word-level timestamps
    print(f"Step 2: Transcribing audio with word timestamps using '{whisper_model_name}' model...")
    model = faster_whisper(whisper_model_name, device="auto", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=language, word_timestamps=True)
    all_words = [word for segment in segments for word in segment.words]
    print("Transcription complete.")

    # 3. Merging diarization and transcription
    print("Step 3: Merging diarization and transcription results...")
    result = []
    speaker_turns = list(diarization.itertracks(yield_label=True))
    
    for word in all_words:
        word_center = (word.start + word.end) / 2
        assigned_speaker = "UNKNOWN"
        for turn, _, speaker in speaker_turns:
            if turn.start <= word_center <= turn.end:
                assigned_speaker = speaker
                break
        result.append({'speaker': assigned_speaker, 'word': word.word, 'start': word.start})

    # 4. Formatting the final transcript
    final_transcript = ""
    current_speaker = None
    current_line = ""
    line_start_time = None

    for res in result:
        if current_speaker != res['speaker']:
            if current_speaker is not None:
                timestamp = str(timedelta(seconds=int(line_start_time))).zfill(8)
                final_transcript += f"[{timestamp}] {current_speaker}:{current_line}\n"
            
            current_speaker = res['speaker']
            line_start_time = res['start']
            current_line = res['word']
        else:
            current_line += res['word']
    
    if current_speaker is not None:
        timestamp = str(timedelta(seconds=int(line_start_time))).zfill(8)
        final_transcript += f"[{timestamp}] {current_speaker}:{current_line}\n"
        
    print("Merging complete.")
    return final_transcript.strip()

def get_prompt(language: str, text: str, is_diarized: bool) -> str:
    """Generates the appropriate prompt for Ollama based on the context."""
    # Prompts remain the same as before
    if is_diarized:
        if language == 'pt':
            return f"""
            Você é um assistente especialista em resumir reuniões a partir de transcrições com identificação de quem falou (diarização).
            A transcrição inclui rótulos como 'SPEAKER_00', 'SPEAKER_01', etc.

            Sua tarefa é criar um resumo conciso e estruturado, em Português do Brasil, e usar os rótulos dos locutores ao atribuir itens.
            Forneça o resumo em três seções:
            1.  **Principais Tópicos Discutidos**: Resumo dos pontos principais.
            2.  **Decisões Tomadas**: Lista de decisões acordadas.
            3.  **Itens de Ação**: Lista de tarefas atribuídas, especificando o locutor responsável (e.g., "SPEAKER_01 precisa enviar o relatório.").

            Transcrição da Reunião:
            ---
            {text}
            ---
            """
        else: # English Diarized
            return f"""
            You are an expert assistant for summarizing meetings from diarized transcripts.
            The transcript includes speaker labels like 'SPEAKER_00', 'SPEAKER_01', etc.

            Your task is to create a concise, structured summary and use the speaker labels when assigning items.
            Provide the summary in three sections:
            1.  **Key Topics Discussed**: Overview of main points.
            2.  **Decisions Made**: Bulleted list of agreed-upon decisions.
            3.  **Action Items**: Bulleted list of assigned tasks, specifying the responsible speaker (e.g., "SPEAKER_01 to send the report.").

            Meeting Transcript:
            ---
            {text}
            ---
            """
    else: # Non-Diarized (original prompts)
        if language == 'pt':
            return f"""
            Você é um assistente especialista em resumir reuniões.
            Sua tarefa é criar um resumo conciso e estruturado da seguinte transcrição de reunião, em Português do Brasil.
            Forneça o resumo em três seções: Principais Tópicos Discutidos, Decisões Tomadas, e Itens de Ação.
            Aqui está a transcrição:\n---\n{text}\n---
            """
        else:
            return f"""
            You are an expert assistant specialized in summarizing meetings.
            Your task is to create a concise and structured summary of the following meeting transcript.
            Provide the summary in three sections: Key Topics Discussed, Decisions Made, and Action Items.
            Here is the transcript:\n---\n{text}\n---
            """

def summarize_text_with_ollama(text: str, model: str, language: str, is_diarized: bool) -> str:
    """Sends the transcript to Ollama to generate a summary."""
    print(f"Sending transcript to Ollama using model '{model}' for summarization...")
    prompt = get_prompt(language, text, is_diarized)
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
    parser = argparse.ArgumentParser(description="Transcribe and summarize a meeting recording with diarization.")
    parser.add_argument("file_path", help="Path to the video or audio file of the meeting.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"The Ollama model to use for summarization (default: {DEFAULT_MODEL}).")
    parser.add_argument("--language", type=str, default='en', help="Language of the meeting audio (e.g., 'en', 'pt', 'es').")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization (requires pyannote.audio).")
    parser.add_argument("--hf_token", type=str, default=None, help="Your Hugging Face Hub token for accessing private models.")
    parser.add_argument("--whisper_model", type=str, default="base", help="The whisper model size to use (e.g., 'tiny', 'base', 'small', 'medium').")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum number of speakers for diarization.")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers for diarization.")
    # The --num_workers argument has been removed.
    args = parser.parse_args()

    if not check_ollama_status(): return

    if args.diarize:
        try:
            import pyannote.audio, torch, faster_whisper
        except ImportError:
            print("\n--- Dependency Error ---\n'--diarize' requires 'pyannote.audio', 'torch', and 'faster-whisper'.\nInstall with: pip install pyannote.audio faster-whisper\n------------------------\n")
            return

    file_path = args.file_path
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found."); return

    file_ext = os.path.splitext(file_path)[1].lower()
    audio_path, is_temp_audio = None, False

    if file_ext in SUPPORTED_AUDIO_FORMATS:
        audio_path = file_path
    else:
        print(f"Error: Unsupported file format '{file_ext}'."); return

    try:
        transcript = ""
        if args.diarize:
            transcript = transcribe_and_diarize(
                audio_path, 
                args.language, 
                args.whisper_model, 
                args.hf_token,
                args.min_speakers,
                args.max_speakers
            )
        else:
            print(f"Transcribing audio with '{args.whisper_model}' model...")
            global faster_whisper
            if faster_whisper is None:
                from faster_whisper import WhisperModel
                faster_whisper = WhisperModel
            model = faster_whisper(args.whisper_model, device="auto", compute_type="int8")
            segments, _ = model.transcribe(audio_path, language=args.language)
            transcript = "\n".join([segment.text for segment in segments])

        print("\n--- Full Transcript ---")
        print(transcript)

        if transcript:
            summary = summarize_text_with_ollama(transcript, args.model, args.language, args.diarize)
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