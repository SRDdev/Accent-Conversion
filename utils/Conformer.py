import torchaudio
from torchaudio.transforms import MelSpectrogram
from torchaudio.models import Conformer
import torch
import time

def extract_conformer_features(file_path):
    start_time = time.time()

    # Load audio file
    waveform, sample_rate = torchaudio.load(file_path)

    # Mel spectrogram transformation
    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=16)
    mel_spectrogram = mel_transform(waveform)

    # Conformer model setup
    input_dim = 80  # The number of mel-frequency spectrogram features
    num_heads = 4
    ffn_dim = 256
    num_layers = 6
    depthwise_conv_kernel_size = 31
    conformer_model = Conformer(
        input_dim=input_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size
    )

    # Remove the final classification layer
    conformer_model = torch.nn.Sequential(*list(conformer_model.children())[:-1])
    conformer_model.eval()

    # Extract features using the Conformer model
    with torch.no_grad():
        features = conformer_model(mel_spectrogram)

    end_time = time.time()
    extraction_time = end_time - start_time

    print("Shape of extracted features:", features.shape)
    print("Extraction Time:", extraction_time, "seconds")

    return features

# Example usage
file_path = 'data/indian/Jerome-ph-non_sanas.wav'
extracted_features = extract_conformer_features(file_path)
