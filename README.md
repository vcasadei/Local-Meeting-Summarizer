# Local-Meeting-Summarizer
A local solution for summarizing online meetings

## Project Goal
To create a command-line tool that takes a audio meeting recording, transcribes it to text, and then uses a local LLM (via Ollama) to generate a summary.

### Feature Upgrade: Speaker Diarization
To add speaker diarization, we will use the powerful `pyannote.audio library`. This requires a few one-time setup steps to get access to the pre-trained models, which are hosted on the [Hugging Face Hub](https://huggingface.co).

## Architecture
The process will follow a simple pipeline:

```
[Audio File] -> [1. Audio Extraction] -> [2. Transcription] -> [3. Summarization] -> [Meeting Summary]
```

1. Audio Extraction: Loads Audio File.

2. Transcription (Speech-to-Text): This is the core of the first phase. We'll use OpenAI's Whisper model. While it's made by OpenAI, Whisper is an open-source model that you can download and run entirely on your local machine. It's highly accurate and has become the industry standard for local transcription.

3. Summarization (LLM): Once we have the full text transcript, we will send it to a local Large Language Model served by Ollama. We'll craft a specific prompt to ask the model to generate a concise summary, identify key decisions, and list action items.

## Setup Instructions
You'll need to set up your environment before running the script.

### Step 1: Install Python
If you don't already have it, install Python 3.8 or newer from the [official Python website](https://www.python.org/downloads/).

### Step 2: Install Ollama
Ollama is the tool that will run the language model locally.

- Go to the [Ollama website](https://ollama.com/) and download the application for your operating system (macOS, Linux, or Windows).

- After installing Ollama, you need to pull a model. We'll use llama3, which is a powerful and versatile model. Open your terminal or command prompt and run:

```shell
ollama run llama3
```

This will download the model and start the Ollama server in the background. You can close the interactive session after it's running.

### Step 3: Install Required Python Libraries
The script depends on a few Python packages. You can install them all with a single `pip` command:

```shell
pip install -r requirements.txt
```

- openai-whisper: For local transcription.
- moviepy: For extracting audio from video files.
- requests: To communicate with the Ollama API.
- tqdm: To show a nice progress bar during transcription, which can take some time.

### Step 4: Hugging Face Hub Setup (for Diarization)

If you plan to use the `--diarize` feature, you need to authenticate with Hugging Face to use the `pyannote` models.

1. **Create a Hugging Face Account**: If you don't have one, create a free account on [huggingface.co/join](https://huggingface.co/join).

2. **Create an Access Token**: Go to your **Settings -> Access Tokens -> New token**. Give it a name (e.g., "pyannote") and the read role. Copy the generated token.

3. **Accept Model User Agreements**: You must visit the pages for the models we'll be using and accept their terms of service. You must be logged into your Hugging Face account.

- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

4. **Provide the Token to the Script**: You have two options to provide your token. The script will use the first one it finds:

- **Option A (Recommended)**: Pass the token directly as a command-line argument. This is the most direct method.

```
--hf_token YOUR_COPIED_TOKEN_HERE
```

- **Option B (Alternative)**: Use the `huggingface-cli login` command. This will cache the token on your machine for the script to use if no token is provided via the command line.

```
huggingface-cli login
```


## Running the script

```shell
# For a video file
python summarize.py "/path/to/your/meeting.mp4"

# For an audio file
python summarize.py "/path/to/your/meeting_audio.mp3"

# To use a different Ollama model (e.g., mistral)
python summarize.py "my_meeting.mp4" --model mistral

# Use the faster engine for transcription
python summarize_fast.py "/path/to/your/meeting.mp4" --fast

# Use Brazilian Portuguese Language
python summarize_fast.py "reuniao_semanal.mp4" --language pt --fast

# Example of diarization for a meeting in Portuguese
python summarize_fast.py "reuniao_com_clientes.mp4" --language pt --diarize

# Example of diarization for a meeting in English
python summarize_fast.py "team_sync.mp4" --language en --diarize

# Example of diarization using a Hugging Face Token directly
python summarize.py "/Users/vcasadei/Documents/GitHub/Local-Meeting-Summarizer/amicorpus/REUNIAO_RH_21082020.mp3" --language pt --diarize --hf_token token --model brunoconterato/Gemma-3-Gaia-PT-BR-4b-it:f16
```

## Next Steps and Potential Improvements

### User Interface: This is a command-line tool. A great next step would be to wrap it in a simple graphical user interface (GUI) using a library like Tkinter or PyQt, or even a web interface using Flask or FastAPI.