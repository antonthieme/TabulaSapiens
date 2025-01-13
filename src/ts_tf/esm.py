import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
import ast


class ProteinDNADataset(Dataset):
    def __init__(self, sequences: pd.Series, pwms: pd.Series, padding: tuple[int, int], max_seq_len: int = 1024):
        self.sequences = sequences # put max sequence length back restriction back?
        self.pwms = [self.process_pwm(pwm) for pwm in pwms]
        self.max_rows = padding[0]
        self.max_cols = padding[1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        pwm = self.pad_pwm(self.pwms[idx])
        return sequence, pwm

    @staticmethod
    def process_pwm(pwm_str: str) -> np.ndarray:
        try:
            list_of_lists = ast.literal_eval(pwm_str)
            return np.array(list_of_lists)
        except Exception as e:
            raise ValueError(f"Error processing PWM: {e}")

    def pad_pwm(self, pwm: np.ndarray) -> np.ndarray:
        padded_pwm = np.zeros((self.max_rows, self.max_cols)) # pad with <pad> token?
        print(f'pwm padded to shape: {padded_pwm.shape} with original shape: {pwm.shape} using max_rows: {self.max_rows} and max_cols: {self.max_cols}')
        padded_pwm[:pwm.shape[0], :pwm.shape[1]] = pwm
        return padded_pwm


class CustomEsmForPWM(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.esm, alphabet = torch.hub.load("facebookresearch/esm", "esm2_t6_8M_UR50D", source="github")
        self.batch_converter = alphabet.get_batch_converter()
        self.output_shape = output_shape
        print(f"Output shape: {output_shape}")
        print(alphabet)
        print(type(alphabet))

        """
        self.classifier = nn.Sequential(
            nn.Linear(self.esm.embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(output_shape)),
            nn.Unflatten(1, output_shape)
        )
        """
        self.classifier = nn.Sequential(
            nn.Linear(self.output_shape[0]-2, 128), # not so sure about this
            nn.ReLU(),
            nn.Linear(128, np.prod(output_shape)),
            nn.Unflatten(1, output_shape)
        )

    def forward(self, sequences: list):
        """
        Forward pass for the model.
        sequences: List of protein sequences (strings).
        """
        _, _, batch_tokens = self.batch_converter([("", seq) for seq in sequences])
        batch_tokens = batch_tokens.to(next(self.parameters()).device)

        # Forward pass through ESM model
        outputs = self.esm(batch_tokens)

        # Use the CLS token embedding (first token) from the final layer
        sequence_embeds = outputs["logits"][:, 0, :]

        # Pass through classifier to predict PWM
        logits = self.classifier(sequence_embeds)
        return logits


    def save_model(self, save_path):
        torch.save(self.state_dict(), f"{save_path}/model_weights.pth")
        torch.save({"output_shape": self.output_shape}, f"{save_path}/model_config.pth")

