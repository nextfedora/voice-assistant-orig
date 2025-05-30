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

- **TTS Model (Piper TTS)**: The script now uses [Piper TTS](https://github.com/rhasspy/piper) for text-to-speech output, falling back to the OS 'say' command if Piper fails or is not configured. Piper voices are generally higher quality and more consistent.
  - `--tts-model`: Specifies the Piper TTS voice. This can be a model name (e.g., `en_US-lessac-medium`) which will be auto-downloaded if not found in `tts-data-dir`, or a direct path to an `.onnx` voice file. A good list of available voices can be found on the [Rhasspy Piper Voices Hugging Face page](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0). The `en_US-lessac-medium` voice is a good starting point for English.
  - `--tts-config`: Optional path to the `.onnx.json` config file for the Piper voice. This is often inferred if `--tts-model` is a path to an `.onnx` file (by looking for a `.json` file with the same name) or if using an auto-downloaded model name.
  - `--tts-data-dir`: Directory where Piper TTS voice models are stored or will be downloaded (default: `./piper_models`).
  - *Note on Voice Quality*: Different Piper voices have varying characteristics, file sizes, and processing requirements. Experiment to find one that best suits your needs for quality and performance.

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

- **Configuring Audio Channels:**
  You can specify the number of audio channels for recording using the `--channels` argument. The default is 1 (mono).
  ```bash
  python main.py --channels 2
  ```
  If you encounter an `OSError: [Errno -9998] Invalid number of channels`, see the Troubleshooting section.

- **Using a specific Piper TTS voice (auto-download):**
  To use a specific Piper TTS voice (e.g., `en_US-lessac-medium`), it will be downloaded to `--tts-data-dir` if not already present.
  ```bash
  python main.py --tts-model en_US-lessac-medium
  ```

- **Using local Piper TTS model files:**
  If you have downloaded Piper model files (`.onnx` and `.onnx.json`) manually:
  ```bash
  python main.py --tts-model /path/to/your/voice.onnx --tts-config /path/to/your/voice.onnx.json
  ```
  You can also place them in the default `--tts-data-dir` (e.g. `./piper_models/en_US-lessac-medium/en_US-lessac-medium.onnx`) and then use the model name.

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
3.  **Run `main.py` with the correct channel count:**
    Use the `maxInputChannels` value for your microphone with the `--channels` argument. For example, if your microphone's `maxInputChannels` is 2, and its index is 0 (which is the default `MIC_IDX`), you would run:
    ```bash
    python main.py --channels 2
    ```
    If your microphone's index is different, you'll need to update `MIC_IDX` in `main.py` accordingly. Common channel values are 1 (mono) or 2 (stereo).
