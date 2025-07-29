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

1. Audio Extraction: If the input is a video file (like a .mp4 from Zoom or Google Meet), we first need to extract the audio track into a format like .mp3. We will use the moviepy library for this. If the input is already an audio file, we can skip this step.

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

### Step 4: (For Linux) Install FFmpeg
The `moviepy` library depends on a system tool called `ffmpeg`.

- On Debian/Ubuntu: `sudo apt update && sudo apt install ffmpeg`

- On Fedora/CentOS: `sudo dnf install ffmpeg`

- macOS and Windows installers for Python libraries often handle this dependency automatically.

### Step 5: Create a Hugging Face Hub Account

`pyannote` requires you to accept user conditions for its models.

1. If you don't have one, create a free account on [huggingface.co](https://huggingface.co/join).

2. Create an access token by going to your **Settings -> Access Tokens -> New token**. Give it a name (e.g., "pyannote") and the `read` role. Copy the token.

### Step 6: Accept Model User Agreements

You must visit the pages for the two models we'll be using and accept their terms of service. You must be logged into your Hugging Face account.

1. Go to [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and agree to the terms.

2. Go to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and agree to the terms.

### Step 7: Log in via your Terminal

The easiest way to give your script access to the models is to log in from your command line.

Run the following command:

```shell
huggingface-cli login
```

It will ask for your token. Paste the token you copied in Step 5 and press Enter.

After completing these four steps, your environment will be ready to run the diarization script. This is a one-time setup.


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
```

## Next Steps and Potential Improvements

### User Interface: This is a command-line tool. A great next step would be to wrap it in a simple graphical user interface (GUI) using a library like Tkinter or PyQt, or even a web interface using Flask or FastAPI.