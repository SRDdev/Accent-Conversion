import threading
import librosa
import sounddevice as sd
import numpy as np
import time
from queue import Queue

#---------------------------Functions---------------------------#
def extract_audio_features(y, sr, feature_queue):
    """
    Extract audio features from the given audio data.

    Parameters:
    - y (numpy.ndarray): Audio data.
    - sr (int): Sampling rate.
    - feature_queue (Queue): A queue to store the extracted features.
    """
    start_time = time.time()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    end_time = time.time()
    response_time = end_time - start_time
    
    # Store features in the queue
    feature_queue.put((chroma_stft, mfccs, spectral_centroid, response_time))

def callback(indata, frames, time, status, feature_queue):
    """
    Callback function for sounddevice stream.

    Parameters:
    - indata (numpy.ndarray): Input audio data.
    - frames (int): Number of frames.
    - time: Timestamp.
    - status: Status of the stream.
    - feature_queue (Queue): A queue to store the extracted features.
    """
    if status:
        print(status)
    if any(indata):
        extract_audio_features(indata[:, 0], 16000, feature_queue)

#---------------------------Streaming---------------------------#
def real_time_audio_processing():
    feature_queue = Queue()

    stop_event = threading.Event()
    stream = sd.InputStream(callback=lambda indata, frames, time, status: callback(indata, frames, time, status, feature_queue), channels=1, samplerate=16000)

    try:
        # Start the stream
        with stream:
            print("Press Ctrl+C to stop the stream")
            while not stop_event.is_set():
                # Check if features are available in the queue
                while not feature_queue.empty():
                    chroma_stft, mfccs, spectral_centroid, response_time = feature_queue.get()
                    print("Chroma STFT shape:", chroma_stft.shape)
                    print("MFCCs shape:", mfccs.shape)
                    print("Spectral Centroid shape:", spectral_centroid.shape)
                    print("Response Time:", response_time, "seconds")

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Set the stop event to stop the processing loop
        stop_event.set()
        # Close the stream
        stream.close()

# Start real-time audio processing
real_time_audio_processing()
