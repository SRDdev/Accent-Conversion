import threading
import librosa
import sounddevice as sd
import numpy as np
import time

#---------------------------Functions---------------------------#
def extract_audio_features(y, sr):
    """
    Extract audio features from the given audio data.

    Parameters:
    - y (numpy.ndarray): Audio data.
    - sr (int): Sampling rate.

    Returns:
    - chroma_stft (numpy.ndarray): Chroma STFT features.
    - mfccs (numpy.ndarray): MFCC features.
    - spectral_centroid (numpy.ndarray): Spectral centroid features.
    - response_time (float): Time taken for feature extraction.
    """
    start_time = time.time()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    end_time = time.time()
    response_time = end_time - start_time
    
    return chroma_stft, mfccs, spectral_centroid, response_time


def callback(indata, frames, time, status):
    """
    Callback function for sounddevice stream.

    Parameters:
    - indata (numpy.ndarray): Input audio data.
    - frames (int): Number of frames.
    - time: Timestamp.
    - status: Status of the stream.
    """
    if status:
        print(status)
    if any(indata):
        chroma_stft, mfccs, spectral_centroid, response_time = extract_audio_features(indata[:, 0], 44100)
        print("Chroma STFT shape:", chroma_stft.shape)
        print("MFCCs shape:", mfccs.shape)
        print("Spectral Centroid shape:", spectral_centroid.shape)
        print("Response Time:", response_time, "seconds")

#---------------------------Streaming---------------------------#
stop_event = threading.Event()
stream = sd.InputStream(callback=callback, channels=1, samplerate=16000)
try:
    # Start the stream
    with stream:
        print("Press Ctrl+C to stop the stream")
        sd.sleep(1000000)
except KeyboardInterrupt:
    print("\nStream stopped by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the stream
    stream.close()
