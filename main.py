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
        self.lock = threading.Lock() # Only for self.generated_text
        self.initial_tts_props = initial_tts_props if initial_tts_props else {}
        # All queue, worker_thread, tts_busy, tts_engine instance, event, startLoop logic removed.

    def on_llm_new_token(self, token, **kwargs):
        with self.lock:
            self.generated_text += token
    
    # process_queue and _on_tts_finish methods are removed.

    def text_to_speech(self, full_text_to_speak):
        if not full_text_to_speak:
            print("No text provided to speak.")
            return

        local_tts_engine = None
        try:
            # print(f"TTS attempting to speak full response (approx {len(full_text_to_speak)} chars).") # Optional: for debugging length
            local_tts_engine = pyttsx3.init()
            if self.initial_tts_props:
                if self.initial_tts_props.get('voice'):
                    local_tts_engine.setProperty('voice', self.initial_tts_props['voice'])
                if self.initial_tts_props.get('rate'):
                    local_tts_engine.setProperty('rate', self.initial_tts_props['rate'])
            
            local_tts_engine.say(full_text_to_speak)
            local_tts_engine.runAndWait()
            print("pyttsx3 playback of full response complete.")
        except Exception as e:
            print(f"Error during pyttsx3 synthesis of full response: {e}")
        # local_tts_engine will be garbage collected.


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
    
    # Initialize pyttsx3 Engine for property gathering
    tts_engine_for_props = None
    initial_tts_props = {} 
    try:
        tts_engine_for_props = pyttsx3.init()
        if args.tts_voice_name and args.tts_voice_name.lower() == 'list':
            voices = tts_engine_for_props.getProperty('voices')
            print("Available TTS voices:")
            for i, voice in enumerate(voices):
                print(f"Voice {i}:")
                print(f"  ID: {voice.id}")
                print(f"  Name: {voice.name}")
                print(f"  Lang: {voice.languages}")
                print(f"  Gender: {voice.gender}")
                print(f"  Age/Rate property (voice.age): {voice.age}") 
                print("-" * 20)
            exit()
        
        # Determine voice ID to use
        if args.tts_voice_name:
            selected_voice_id = None
            voices = tts_engine_for_props.getProperty('voices')
            for voice in voices:
                if voice.name and args.tts_voice_name.lower() == voice.name.lower():
                    selected_voice_id = voice.id
                    break
            if selected_voice_id:
                initial_tts_props['voice'] = selected_voice_id
                print(f"TTS voice for use: {args.tts_voice_name} (ID: {selected_voice_id})")
            else:
                print(f"TTS voice name '{args.tts_voice_name}' not found. Using default system voice ID.")
                initial_tts_props['voice'] = tts_engine_for_props.getProperty('voice') # Store default ID
        else:
            initial_tts_props['voice'] = tts_engine_for_props.getProperty('voice') # Store default ID if no name given

        initial_tts_props['rate'] = args.tts_rate 
        print(f"TTS rate for use: {args.tts_rate}")
        # tts_engine_for_props is temporary and can be garbage collected.

    except Exception as e:
        print(f"Error initializing pyttsx3 for property gathering: {e}. Default voice/rate may be used by handler.")
        if not initial_tts_props.get('rate'): 
           initial_tts_props['rate'] = args.tts_rate # Default from argparse if engine failed early
        # initial_tts_props['voice'] will be None if engine failed, handler's local engine will use system default.
     

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
            # tts_busy flag is removed, main loop no longer needs to check it.
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
            # Clear previous full response from handler before new LLM call
            with voice_output_handler.lock:
                voice_output_handler.generated_text = ""

            prompt = prompt_template.format(dialogue=dialogue + "*Q* {}\n".format(user_input)) # Add current Q to prompt
            reply = llm(prompt, max_tokens=4096) # LLM call

            if reply is not None: 
                # The 'generated_text' in handler has been accumulating this 'reply' via on_llm_new_token.
                # We use the text from the handler as the single source of truth for what was generated.
                full_response_text = ""
                with voice_output_handler.lock: # Access generated_text safely
                    full_response_text = voice_output_handler.generated_text
                    # voice_output_handler.generated_text = "" # Reset for next turn is done before LLM call now
                
                dialogue += "*Q* {}\n*A* {}\n".format(user_input, full_response_text) # Add Q&A to dialogue history
                print("%s: %s (Time %d ms)" % ("Server", full_response_text.strip(), (time.time() - time_ckpt) * 1000))
                
                if full_response_text.strip(): # Only speak if there's text
                    voice_output_handler.text_to_speech(full_response_text.strip()) 
            else: # Handle case where LLM returns None (e.g. error or empty response)
                dialogue += "*Q* {}\n*A* \n".format(user_input) # Log Q with empty A
                print("%s: %s (Time %d ms)" % ("Server", "[No reply from LLM]", (time.time() - time_ckpt) * 1000))

    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt...")
    finally:
        # No tts_engine.endLoop() needed as the handler uses local, short-lived engines.
        # The tts_engine_for_props is also short-lived and doesn't run a loop.
        # PyAudio for input is managed within record_audio().
        print("Script finished.")
