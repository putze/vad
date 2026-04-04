import torch

from src.vad.data.datasets.librivad import LibriVADDataset
from src.vad.data.loaders import build_train_loader
from src.vad.data.preprocessing.audio import AudioPreprocessor
from src.vad.data.preprocessing.features import LogMelFeatureExtractor
from src.vad.data.preprocessing.labels import LabelAligner
from src.vad.data.preprocessing.preprocessing import VADPreprocessor
from src.vad.models.causal_conv import CausalVAD


def build():
    audio_preprocessor = AudioPreprocessor(
        target_sample_rate=16000,
    )

    feature_extractor = LogMelFeatureExtractor(
        sample_rate=16000, n_mels=40, n_fft=400, hop_length=160, frame_length=400, center=False
    )

    label_aligner = LabelAligner(
        hop_length=160,
        frame_length=400,
        center=False,
    )

    processor = VADPreprocessor(
        audio_preprocessor=audio_preprocessor,
        feature_extractor=feature_extractor,
        label_aligner=label_aligner,
    )

    raw_train_dataset = LibriVADDataset(
        results_root="/Users/antje/Blynt/LibriVAD/Results",
        labels_root="/Users/antje/Blynt/LibriVAD/Files/Labels",
        datasets=["LibriSpeech"],
        splits=["train-clean-100"],
    )

    train_loader = build_train_loader(
        raw_dataset=raw_train_dataset, processor=processor, batch_size=16, num_workers=0
    )

    model = CausalVAD(n_mels=40)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for x, y, lengths in train_loader:
        logits = model(x)  # [B, 2, T]
        loss = criterion(logits, y)
        print(logits.shape)
        print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    build()
