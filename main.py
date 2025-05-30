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
    def __init__(self, pyttsx3_engine_instance=None):
        self.generated_text = ""
        self.lock = threading.Lock() # For generated_text and speech_queue
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.process_queue)
        self.worker_thread.daemon = True
        
        self.tts_engine = pyttsx3_engine_instance
        self.tts_loop_started = False
        self.tts_utterance_finished_event = threading.Event()
        self.current_utterance_name = "u_tts" # Static name for utterances

        if self.tts_engine:
            try:
                self.tts_engine.connect('finished-utterance', self._on_tts_finish)
                self.tts_engine.startLoop(False) # Prepare for external event iteration
                self.tts_loop_started = True
                print("pyttsx3 event loop started (external management) and 'finished-utterance' callback connected.")
            except Exception as e:
                print(f"Error starting pyttsx3 external loop or connecting callback: {e}. TTS will be non-functional.")
                self.tts_engine = None # Cannot use if loop or callback setup failed
        
        self.worker_thread.start() # Start worker thread after engine is potentially set up
        self.tts_busy = False

    def _on_tts_finish(self, name, completed):
        # print(f"TTS event: name='{name}', completed={completed}, expected='{self.current_utterance_name}'") # Debugging
        if name == self.current_utterance_name:
            self.tts_utterance_finished_event.set()
        # else:
            # This might indicate an issue if events from other sources or old utterances arrive
            # print(f"TTS event for unexpected utterance name: {name}. Current expected: {self.current_utterance_name}")

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
            self.text_to_speech(text) # This will now block via the iterate loop in text_to_speech
            self.speech_queue.task_done()
            if self.speech_queue.empty():
                self.tts_busy = False

    def text_to_speech(self, text):
        if self.tts_engine and self.tts_loop_started:
            try:
                self.tts_utterance_finished_event.clear()
                print(f"Queuing with pyttsx3 (event callback): {text}")
                self.tts_engine.say(text, self.current_utterance_name)
                
                # print(f"Starting pyttsx3 iteration for '{self.current_utterance_name}'...") # Debugging
                while not self.tts_utterance_finished_event.is_set():
                    self.tts_engine.iterate()
                    time.sleep(0.01) # Prevent tight loop, yield CPU
                # print(f"pyttsx3 utterance '{self.current_utterance_name}' finished processing (event received).") # Debugging
            except Exception as e:
                print(f"Error during pyttsx3 synthesis or iteration (event callback): {e}")
                # Ensure event is set in case of error to prevent deadlocks
                self.tts_utterance_finished_event.set() 
        elif not self.tts_engine:
            print("pyttsx3 engine not initialized in handler. Cannot speak.")
        elif not self.tts_loop_started:
            print("pyttsx3 external loop not started in handler. Cannot speak.")


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
                # No need to call tts_engine.stop() or similar before exit for list voices
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

    except Exception as e:
        print(f"Error initializing pyttsx3: {e}. TTS might not be functional.")
        tts_engine = None # Ensure tts_engine is None if setup failed


    if LANG == "CN":
        prompt_path = "prompts/example-cn.txt"
    else:
        prompt_path = "prompts/example-en.txt"
    with open(prompt_path, 'r', encoding='utf-8') as file:
        template = file.read().strip() # {dialogue}
    prompt_template = PromptTemplate(template=template, input_variables=["dialogue"])

    # Create an instance of the VoiceOutputCallbackHandler
    # Pass the initialized and configured tts_engine
    voice_output_handler = VoiceOutputCallbackHandler(pyttsx3_engine_instance=tts_engine)

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
        if 'voice_output_handler' in locals() and hasattr(voice_output_handler, 'tts_engine') and \
           voice_output_handler.tts_engine is not None and \
           hasattr(voice_output_handler, 'tts_loop_started') and voice_output_handler.tts_loop_started:
            try:
                print("Stopping pyttsx3 event loop.")
                voice_output_handler.tts_engine.endLoop()
            except Exception as e:
                print(f"Error stopping pyttsx3 event loop: {e}")
        
        # PyAudio for input is managed within record_audio()
        print("Script finished.")
