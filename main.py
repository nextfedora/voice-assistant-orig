import argparse
import time
import wave
import queue
import struct
import threading
import subprocess

import pyaudio
import whisper
import pyttsx3

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
# MIC_IDX = 0 # Replaced by --mic-device-index arg

def compute_rms(data):
    # Assuming data is in 16-bit samples
    format = "<{}h".format(len(data) // 2)
    ints = struct.unpack(format, data)

    # Calculate RMS
    sum_squares = sum(i ** 2 for i in ints)
    rms = (sum_squares / len(ints)) ** 0.5
    return rms

def record_audio(num_channels, device_idx):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=num_channels, rate=RATE, input=True, input_device_index=device_idx, frames_per_buffer=CHUNK)

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
    def __init__(self, initial_tts_props=None):
        self.generated_text = ""
        self.lock = threading.Lock()
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.tts_busy = False
        self.initial_tts_props = initial_tts_props

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
        local_tts_engine = None
        try:
            # Initialize a new engine instance for each call
            local_tts_engine = pyttsx3.init()
            if self.initial_tts_props:
                if self.initial_tts_props.get('voice'):
                    local_tts_engine.setProperty('voice', self.initial_tts_props['voice'])
                if self.initial_tts_props.get('rate'):
                    local_tts_engine.setProperty('rate', self.initial_tts_props['rate'])
            
            print(f"Synthesizing with pyttsx3: {text}")
            local_tts_engine.say(text)
            local_tts_engine.runAndWait()
            print("pyttsx3 playback complete.")
            # No explicit stop needed for local_tts_engine, it will be garbage collected.
            # However, some pyttsx3 drivers might have issues with rapid re-initialization.
            # If issues persist, a single engine in process_queue with stop/start iterations might be needed.
        except RuntimeError as r_err:
            # Catching RuntimeError specifically for "run loop already started"
            print(f"pyttsx3 runtime error in text_to_speech: {r_err}. This might indicate an issue with the TTS engine driver even with local instances.")
        except Exception as e:
            print(f"General error during pyttsx3 synthesis or playback in text_to_speech: {e}")


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
    # Removed Piper TTS arguments: --tts-model, --tts-config, --tts-data-dir
    parser.add_argument(
        "--tts-voice-name",
        type=str,
        default=None, # Uses pyttsx3 default voice if not specified
        help="Name of the pyttsx3 voice to use (e.g., 'Alex', 'Samantha' on macOS). Use 'list' to see available voices."
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        default=180, # Default rate
        help="Speech rate for TTS (words per minute)."
    )
    parser.add_argument(
        "--mic-device-index",
        type=int,
        default=0, 
        help="Device index of the microphone to use for recording. Use tools/list_microphones.py to find the correct index."
    )
    args = parser.parse_args()
    
    # Initialize pyttsx3 Engine
    tts_engine = None
    initial_tts_props = {} # Default to empty dict

    try:
        tts_engine = pyttsx3.init()

        if args.tts_voice_name:
            if args.tts_voice_name.lower() == 'list':
                voices = tts_engine.getProperty('voices')
                print("Available TTS voices:")
                for i, voice in enumerate(voices):
                    print(f"Voice {i}:")
                    print(f"  ID: {voice.id}")
                    print(f"  Name: {voice.name}")
                    print(f"  Lang: {voice.languages}")
                    print(f"  Gender: {voice.gender}")
                    print(f"  Age/Rate property (voice.age): {voice.age}") 
                    print("-" * 20)
                # Safe to exit, no resources needing explicit cleanup beyond OS handling for pyttsx3
                exit() 
            
            selected_voice_id = None
            voices = tts_engine.getProperty('voices')
            for voice in voices:
                if voice.name and args.tts_voice_name.lower() == voice.name.lower():
                    selected_voice_id = voice.id
                    break
            if selected_voice_id:
                tts_engine.setProperty('voice', selected_voice_id)
                print(f"TTS voice set to: {args.tts_voice_name}")
            else:
                print(f"TTS voice name '{args.tts_voice_name}' not found. Using default system voice.")
        
        # Always set rate, using default if not specified by user
        tts_engine.setProperty('rate', args.tts_rate)
        print(f"TTS rate set to: {args.tts_rate}")

        # Store properties for the callback handler
        initial_tts_props['voice'] = tts_engine.getProperty('voice')
        initial_tts_props['rate'] = tts_engine.getProperty('rate')

    except Exception as e:
        print(f"Error initializing pyttsx3 or getting properties: {e}. TTS might not work or use system defaults.")
        # tts_engine might be None or partially configured.
        # initial_tts_props will use defaults or be empty if error before props could be read.
        # If tts_engine is None here, initial_tts_props remains empty, callback will use pyttsx3 defaults.
        if args.tts_rate and not initial_tts_props.get('rate'): # Ensure rate from args is respected if engine init failed late
            initial_tts_props['rate'] = args.tts_rate


    if LANG == "CN":
        prompt_path = "prompts/example-cn.txt"
    else:
        prompt_path = "prompts/example-en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip() # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])

    # Create an instance of the VoiceOutputCallbackHandler
    voice_output_handler = VoiceOutputCallbackHandler(initial_tts_props=initial_tts_props)

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
                record_audio(args.channels, args.mic_device_index)
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
        # PyAudio for input is managed within record_audio()
        # No global PyAudio output instance for pyttsx3 to terminate here
        print("Script finished.")
