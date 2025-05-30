# A Simple Voice Assistant Script

English | [简体中文](README-CN.md)

This is a simple Python script project that allows dialogue with a local large language model through voice.

The voice recognition part of this project is from the [Apple MLX example repo](https://github.com/ml-explore/mlx-examples/tree/main/whisper), and the textual responses are generated using the Yi model from [01.AI](https://www.lingyiwanwu.com). For more details, see the [Acknowledgments](## Acknowledgments) section.

### File Structure

```bash
├───main.py
├───models
├───prompts
├───recordings
├───tools
│   └───list_microphones.py
├───whisper
```

This project is a single-script project, with main.py containing all program logic. The `models/` folder stores model files. `prompts/` contains prompt words. `recordings/` holds temporary recordings. `tools/list_microphones.py` is a simple script to view the microphone list, used in `main.py` to specify the microphone number. `whisper/` is from the [Apple MLX example repo](https://github.com/ml-explore/mlx-examples/tree/main/whisper), used for recognizing user's voice input.

## Installation Guide

This project is based on the Python programming language, and the Python version used for program operation is 3.11.5. It is recommended to configure the Python environment using [Anaconda](https://www.anaconda.com). The following setup process has been tested and passed on macOS systems. Windows and Linux can use speech_recognition and pyttsx3 to replace the whisper and say commands mentioned below. The following are console/terminal/shell commands.

### Environment Configuration
```
conda create -n VoiceAI python=3.11
conda activate VoiceAI
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Install audio processing tools
# On macOS (using Homebrew):
brew install portaudio ffmpeg
# On Linux (Debian/Ubuntu):
# sudo apt-get install portaudio19-dev ffmpeg
# For other systems, please refer to the official websites for PortAudio and FFmpeg.

pip install pyaudio
```

### Model Files
The model files are stored in the `models/` folder. The paths to the models can be specified using command-line arguments when running `main.py`.

- **LLM Model**: Specified by the `--llm-model` argument.
  - Default: `models/yi-34b-chat.Q8_0.gguf`
  - It is recommended to download gguf format models from TheBloke and XeIaso. The 6B model has a smaller memory footprint:
    - [TheBloke/Yi-34B-Chat-GGUF](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q8_0.gguf)
    - [XeIaso/Yi-6B-Chat-GGUF](https://huggingface.co/XeIaso/yi-chat-6B-GGUF/blob/main/yi-chat-6b.Q8_0.gguf)
- **Whisper Model**: Specified by the `--whisper-model` argument.
  - Default: `models/whisper-large-v3/`
  - The [version for MLX](https://huggingface.co/mlx-community/whisper-large-v3-mlx) from `mlx-community` can be directly downloaded.
  - For a smaller footprint, you can use models like [mlx-community/whisper-tiny-mlx](https://huggingface.co/mlx-community/whisper-tiny-mlx).

- **Text-to-Speech (TTS) Voice (pyttsx3)**: The script uses `pyttsx3` for text-to-speech output. `pyttsx3` is a cross-platform library that interfaces with available system TTS engines (e.g., NSSpeechSynthesizer on macOS, SAPI5 on Windows, eSpeak on Linux).
  - `--tts-voice-name`: Specifies the name of the voice to be used by `pyttsx3`. The available voices depend on the operating system and installed voice packages (e.g., 'Alex', 'Samantha' on macOS). If the name is not found or not specified, the system's default voice will be used. Provide `'list'` as the value to print all available voice names and their details, then exit (e.g., `python main.py --tts-voice-name list`).
  - `--tts-rate`: Sets the speech rate for TTS in words per minute (default: 180).
  - *Improving Voice Quality*: On macOS, you can install additional voices in System Settings > Accessibility > Spoken Content > System Voice > Manage Voices. These voices should then become available for use with the `--tts-voice-name` argument. Similar options may be available on other operating systems.

- **Audio Input Configuration**:
  - `--channels`: Integer number of audio channels for recording (default: `1`). Common values are 1 (mono) or 2 (stereo).
  - `--mic-device-index`: Integer device index of the microphone to use for recording (default: `0`). Use the `tools/list_microphones.py` script to find the correct index for your microphone.

### Running with Specific Models

You can specify which models to use when running the script from the command line.

- **Using default models:**
  If you run the script without any model arguments, it will use the default paths:
  ```bash
  python main.py
  ```

- **Using a smaller LLM:**
  To use a different LLM, such as the smaller 6B Yi model, use the `--llm-model` argument:
  ```bash
  python main.py --llm-model models/yi-chat-6b.Q8_0.gguf
  ```
  Make sure you have downloaded this model into your `models/` directory or provide the correct path.

- **Using a smaller Whisper model:**
  To use a smaller Whisper model, like `whisper-tiny-mlx`, download it from [mlx-community/whisper-tiny-mlx](https://huggingface.co/mlx-community/whisper-tiny-mlx) and place it in your `models/` directory (e.g., `models/whisper-tiny-mlx`). Then run:
  ```bash
  python main.py --whisper-model models/whisper-tiny-mlx
  ```

- **Using both smaller LLM and Whisper models:**
  You can combine these arguments:
  ```bash
  python main.py --llm-model models/yi-chat-6b.Q8_0.gguf --whisper-model models/whisper-tiny-mlx
  ```

- **Configuring Audio Input (Microphone and Channels):**
  If the default microphone (index 0) or channel count (1) is not suitable, you can specify them. For example, to use microphone index 2 with 1 channel:
  ```bash
  python main.py --mic-device-index 2 --channels 1
  ```
  If you encounter an `OSError: [Errno -9998] Invalid number of channels` with your chosen microphone, see the Troubleshooting section.

- **Listing available TTS voices:**
  To see which voices are available on your system for `pyttsx3` to use:
  ```bash
  python main.py --tts-voice-name list
  ```

- **Using a specific TTS voice and rate:**
  Once you know a voice name (e.g., "Alex", "Samantha" on macOS, or others depending on your system), you can use it:
  ```bash
  python main.py --tts-voice-name Alex --tts-rate 200
  ```
  If the specified voice name is not found, the script will use the default system voice.

#### Original Model Files Acknowledgement

The voice recognition part of this project is based on OpenAI's whisper model; its implementation comes from the [Apple MLX example repo](https://github.com/ml-explore/mlx-examples/tree/main/whisper). The version used in this project is from January 2024, #80d1867. In the future, users can fetch new versions as needed.

The responses in this project are generated by the large language model Yi from [01.AI](https://www.lingyiwanwu.com), where Yi-34B-Chat is more powerful. The [8-bit quantized version made by TheBloke](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF) has a memory footprint of 39.04 GB and is recommended for use if hardware conditions permit. This model runs locally based on the [LangChain](https://www.langchain.com) framework and [llama.cpp by Georgi Gerganov](https://github.com/ggerganov/llama.cpp).

Thank you to all selfless programmers for their contributions to the open-source community!

## Troubleshooting

### OSError: [Errno -9998] Invalid number of channels

This error indicates that the number of audio channels specified (or defaulted to) for recording is not supported by your selected microphone.

1.  **List available microphones and their capabilities:**
    Run the `tools/list_microphones.py` script:
    ```bash
    python tools/list_microphones.py
    ```
2.  **Identify your microphone:**
    Look for your intended microphone in the output. Note its `index` and `maxInputChannels`.
3.  **Run `main.py` with the correct microphone index and channel count:**
    Use the `index` value from the tool as the argument for `--mic-device-index`, and the `maxInputChannels` value for the `--channels` argument. For example, if your microphone's index is `2` and its `maxInputChannels` is `1`:
    ```bash
    python main.py --mic-device-index 2 --channels 1
    ```
    Common channel values are 1 (mono) or 2 (stereo).
