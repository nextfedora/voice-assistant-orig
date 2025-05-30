import argparse
import time
import wave
import queue
import struct
import threading
import subprocess
import io

import pyaudio
import whisper
import soundfile as sf
# Attempt to import PiperVoice, hoping the actual library uses this name or similar
# If this fails at runtime, the user will need to install piper_tts and check its API
from piper_tts.piper_voice import PiperVoice 

from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

LANG = "EN" # CN for Chinese, EN for English
DEBUG = True

# Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
# CHANNELS = 1 # Will be replaced by args.channels
RATE = 44100
SILENCE_THRESHOLD = 500
SILENT_CHUNKS = 2 * RATE / CHUNK  # two seconds of silence marks the end of user voice input
MIC_IDX = 0 # Set microphone id. Use tools/list_microphones.py to see a device list.

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio(num_channels):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=num_channels, rate=RATE, input=True, input_device_index=MIC_IDX, frames_per_buffer=CHUNK)

    silent_chunks = 0
    audio_started = False
    frames = []

    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = compute_rms(data)
        if audio_started:
            if rms < SILENCE_THRESHOLD:
                silent_chunks += 1
                if silent_chunks > SILENT_CHUNKS:
                    break
            else:
                silent_chunks = 0
        elif rms >= SILENCE_THRESHOLD:
            audio_started = True

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save audio to a WAV file
    with wave.open('recordings/output.wav', 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

class VoiceOutputCallbackHandler(BaseCallbackHandler):
    def __init__(self, piper_voice_instance=None, pyaudio_instance=None):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False
        self.piper_voice = piper_voice_instance
        self.p_audio = pyaudio_instance

    def on_llm_new_token(self, token, **kwargs):
        # Append the token to the generated text
        with self.lock:
            self.generated_text += token

        # Check if the token is the end of a sentence
        if token in ['.', '。', '!', '！', '?', '？']:
            with self.lock:
                # Put the complete sentence in the queue
                self.speech_queue.put(self.generated_text)
                self.generated_text = ""

    def process_queue(self):
        while True:
            # Wait for the next sentence
            text = self.speech_queue.get()
            if text is None:
                self.tts_busy = False
                continue
            self.tts_busy = True
            self.text_to_speech(text)
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        if self.piper_voice and self.p_audio:
            try:
                print(f"Synthesizing with Piper: {text}")
                # Assumed API for Piper TTS synthesis. The actual method might differ.
                # It's expected to return WAV audio bytes.
                wav_bytes = self.piper_voice.synthesize(text) 

                if wav_bytes:
                    # Use soundfile to get properties from WAV bytes for PyAudio
                    data, samplerate = sf.read(io.BytesIO(wav_bytes))
                    
                    # Open PyAudio stream for playback
                    # Piper models are typically mono, soundfile usually returns float32
                    stream = self.p_audio.open(format=pyaudio.paFloat32, 
                                               channels=1, 
                                               rate=samplerate,
                                               output=True)
                    # Play audio
                    stream.write(data.astype('float32').tobytes()) # Ensure data is in bytes
                    stream.stop_stream()
                    stream.close()
                    print("Piper TTS playback complete.")
                else:
                    print("Piper TTS synthesis returned no data. Falling back.")
                    self.fallback_tts(text) # Fallback if Piper returns no data
            except Exception as e:
                print(f"Error during Piper TTS synthesis or playback: {e}. Falling back.")
                self.fallback_tts(text) # Fallback on any Piper error
        else:
            # Fallback if Piper or PyAudio not initialized
            self.fallback_tts(text)

    def fallback_tts(self, text):
        print(f"Falling back to OS 'say' command for: {text}")
        try:
            if LANG == "CN":
                subprocess.call(["say", "-r", "200", "-v", "TingTing", text])
            else:
                subprocess.call(["say", "-r", "180", "-v", "Karen", text])
        except Exception as e:
            print(f"Error in fallback text-to-speech: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Voice Assistant with configurable models.")
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="models/whisper-large-v3",
        help="Path to the Whisper model."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="models/yi-34b-chat.Q8_0.gguf", # Or models/yi-chat-6b.Q8_0.gguf
        help="Path to the LLM model."
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of audio channels for recording."
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="en_US-lessac-medium",
        help="Piper TTS voice model name (e.g., en_US-lessac-medium) or path to .onnx file. If a name, it will try to download. See Piper docs for voice names."
    )
    parser.add_argument(
        "--tts-config",
        type=str,
        default=None,
        help="Path to Piper TTS .onnx.json config file. If --tts-model is a name, this can often be inferred."
    )
    parser.add_argument(
        "--tts-data-dir",
        type=str,
        default="./piper_models",
        help="Directory to find/download Piper voice models."
    )
    args = parser.parse_args()

    # Initialize Piper TTS Voice
    piper_voice = None
    try:
        print(f"Initializing Piper TTS with model: {args.tts_model}, config: {args.tts_config}, data directory: {args.tts_data_dir}")
        # This instantiation assumes piper-tts can handle a model name for download,
        # or direct paths if tts_model is a path to .onnx and tts_config is its .json.
        # The actual API might require specific handling for names vs paths.
        piper_voice = PiperVoice(model_name_or_path=args.tts_model, config_path=args.tts_config, data_folder=args.tts_data_dir)
        print("Piper TTS initialized successfully.")
    except Exception as e:
        print(f"Error initializing Piper TTS: {e}. Ensure 'piper-tts' is installed and models are accessible.")
        print("TTS will fall back to OS 'say' command if available, or fail if 'say' is not available.")
        piper_voice = None # Ensure it's None if init fails

    # Initialize PyAudio instance for playback
    p_audio_out = pyaudio.PyAudio()

    if LANG == "CN":
        prompt_path = "prompts/example-cn.txt"
    else:
        prompt_path = "prompts/example-en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip() # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler(piper_voice_instance=piper_voice, pyaudio_instance=p_audio_out)

    # Create a callback manager with the voice output handler
    callback_manager = BaseCallbackManager(handlers=[voice_output_handler])

    llm = LlamaCpp(
        model_path=args.llm_model,
        n_gpu_layers=1, # Metal set to 1 is enough.
        n_batch=512,    # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        n_ctx=4096,     # Update the context window size to 4096
        f16_kv=True,    # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        stop=["<|im_end|>"],
        verbose=False,
    )
    dialogue = ""
    try:
        while True:
            if voice_output_handler.tts_busy:  # Check if TTS is busy
                continue  # Skip to the next iteration if TTS is busy 
            try:
                print("Listening...")
                record_audio(args.channels)
                print("Transcribing...")
                time_ckpt = time.time()
                user_input = whisper.transcribe("recordings/output.wav", path_or_hf_repo=args.whisper_model)["text"]
                print("%s: %s (Time %d ms)" % ("Guest", user_input, (time.time() - time_ckpt) * 1000))
            
            except subprocess.CalledProcessError:
                print("voice recognition failed, please try again")
                continue
            time_ckpt = time.time()
            print("Generating...")
            dialogue += "*Q* {}\n".format(user_input)
            prompt = prompt_template.format(dialogue=dialogue)
            reply = llm(prompt, max_tokens=4096)
            if reply is not None:
                voice_output_handler.speech_queue.put(None)
                dialogue += "*A* {}\n".format(reply)
                print("%s: %s (Time %d ms)" % ("Server", reply.strip(), (time.time() - time_ckpt) * 1000))
    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
    finally:
        if 'p_audio_out' in locals() and p_audio_out is not None:
            print("Terminating PyAudio for output.")
            p_audio_out.terminate()
        # Terminate the input PyAudio instance used in record_audio if it were managed globally
        # However, record_audio() creates and terminates its own PyAudio instance locally, so no global one to clean up here for input.
