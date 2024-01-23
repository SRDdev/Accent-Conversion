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


