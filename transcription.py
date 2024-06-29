from numpy import record
import pyaudio
import wave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def record_audio(output_filename, duration=5):
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    rate = 44100
    record_seconds = duration

    p = pyaudio.PyAudio()

    print("Recording")

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=rate,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = []

    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished recording")

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

def transcribe_audio():
    audio = record_audio("output.wav", duration=10)
    # Need OPENAI_API_KEY in .env or pass it in
    client = OpenAI()
    audio_file = open("output.wav", "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript
