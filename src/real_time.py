from Real_time_extractor import *
from Real_time_vocoder import *
import threading 

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




import torchaudio
from speechbrain.pretrained import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from queue import Queue
import numpy as np

def real_time_vocoder(feature_queue):
    # Load a pretrained HIFIGAN Vocoder
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="vocoder_16khz")

    while True:
        # Check if features are available in the queue
        while not feature_queue.empty():
            chroma_stft, mfccs, spectral_centroid, _ = feature_queue.get()

            # Combine features as needed for vocoder
            combined_features = np.concatenate((chroma_stft, mfccs, spectral_centroid), axis=0)

            # Perform vocoding
            # For demonstration purposes, use the first channel of the combined features
            signal = combined_features[0]

            # Ensure the audio is single channel
            signal = signal.squeeze()

            # Compute the mel spectrogram.
            # IMPORTANT: Use these specific parameters to match the Vocoder's training settings for optimal results.
            spectrogram, _ = mel_spectogram(
                audio=signal.squeeze(),
                sample_rate=16000,
                hop_length=256,
                win_length=1024,
                n_mels=80,
                n_fft=1024,
                f_min=0.0,
                f_max=8000.0,
                power=1,
                normalized=False,
                min_max_energy_norm=True,
                norm="slaney",
                mel_scale="slaney",
                compression=True
            )

            # Convert the spectrogram to waveform
            waveforms = hifi_gan.decode_batch(spectrogram)

            # Save the reconstructed audio as a waveform
            torchaudio.save('waveform_reconstructed.wav', waveforms.squeeze(1), 16000)






# Start real-time vocoding
feature_queue = Queue()
vocoder_thread = threading.Thread(target=real_time_vocoder, args=(feature_queue,))
vocoder_thread.start()

# Real-time audio processing is handled in Code 1, no modifications needed there.
# Ensure both threads run concurrently for real-time processing.
real_time_audio_processing()