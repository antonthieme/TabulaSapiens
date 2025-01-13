import argparse
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from pathlib import Path
import sys
import numpy as np
import ast
from torch.utils.tensorboard import SummaryWriter

# Add src_path to sys.path
src_path = Path("../src")
sys.path.append(str(src_path))
src_path = Path("../../src")
sys.path.append(str(src_path))

from ts_tf.esm import ProteinDNADataset, CustomEsmForPWM

writer = SummaryWriter()

def setup_logging(log_file: str):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_pwm(pwm_str: str) -> np.ndarray:
    try:
        list_of_lists = ast.literal_eval(pwm_str)
        return np.array(list_of_lists)
    except Exception as e:
        raise ValueError(f"Error processing PWM: {e}")

def fine_tune_esm(csv_file: str, epochs: int = 3, lr: float = 2e-5, k_folds: int = 5):
    """Fine-tune the ESM model for protein-DNA interaction prediction."""
    try:
        logging.info("Starting fine-tuning process.")

        # Load data
        data = pd.read_csv(csv_file)
        sequences = data['AA Sequence']
        pwms = data['pwm']
        print(data)

        pwms_decoded = [process_pwm(pwm) for pwm in pwms]

        max_rows = max(pwm.shape[0] for pwm in pwms_decoded)
        max_cols = max(pwm.shape[1] for pwm in pwms_decoded)

        # Initialize dataset and model
        example_dataset = ProteinDNADataset(sequences, pwms, padding=(max_rows, max_cols)) # why defined here and then train and validation again
        print(f'example_dataset.max_rows: {example_dataset.max_rows}')
        print(f'example_dataset.max_cols: {example_dataset.max_cols}')
        model = CustomEsmForPWM(output_shape=(example_dataset.max_rows, example_dataset.max_cols))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # K-Fold Cross Validation
        kfold = KFold(n_splits=k_folds, shuffle=True)
        for fold, (train_ids, val_ids) in enumerate(kfold.split(sequences)):
            logging.info(f"Starting Fold {fold + 1}/{k_folds}")

            # Prepare data loaders
            sequences = data['AA Sequence']
            pwms = data['pwm']
            print(f'type of sequences: {type(sequences)}')
            print(f'sequences: {sequences}')
            print(f'type of pwms: {type(pwms)}')
            print(f'pwms: {pwms}')
            train_sequences = sequences.iloc[train_ids]
            train_pwms = pwms.iloc[train_ids]
            val_sequences = sequences.iloc[val_ids]
            val_pwms = pwms.iloc[val_ids]

            train_dataset = ProteinDNADataset(train_sequences, train_pwms, padding=(max_rows, max_cols))
            val_dataset = ProteinDNADataset(val_sequences, val_pwms, padding=(max_rows, max_cols))
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch in train_dataloader:
                    try:
                        sequences, pwms = batch
                        sequences = list(sequences)
                        pwms = torch.tensor(pwms, dtype=torch.float).to(next(model.parameters()).device)

                        optimizer.zero_grad()
                        outputs = model(sequences)
                        print(f'training pwms shape: {pwms.shape}')
                        print(f'training outputs shape: {outputs.shape}')
                        loss = torch.nn.functional.mse_loss(outputs, pwms)
                        writer.add_scalar("Loss/train", loss, epoch)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                    except Exception as e:
                        logging.error(f"Error during training batch: {e}")
                        raise

                logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader):.4f}")

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        try:
                            sequences, pwms = val_batch
                            sequences = list(sequences)
                            pwms = torch.tensor(pwms, dtype=torch.float).to(next(model.parameters()).device)
                            print(f'validation pwms shape: {pwms.shape}')
                            outputs = model(sequences)
                            print(f'validation outputs shape: {outputs.shape}')
                            val_loss += torch.nn.functional.mse_loss(outputs, pwms).item()
                        except Exception as e:
                            logging.error(f"Error during validation batch: {e}")
                            raise

                val_loss /= len(val_dataloader)
                logging.info(f"Validation Loss: {val_loss:.4f}")

        # Save model
        model.save_model('./fine_tuned_esm')
        logging.info("Model saved to './fine_tuned_esm'")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"Empty or invalid CSV file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise

def run_fine_tune_esm(csv_file: str):
    """Run fine-tuning process."""

    try:
        fine_tune_esm(csv_file, epochs=3)
    except Exception as e:
        logging.critical(f"Fine-tuning process failed: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune ESM for protein-DNA interaction prediction.")
    parser.add_argument('csv_file', type=str, help='CSV file containing amino acid sequences and PWMs')
    parser.add_argument('log_file', type=str, help='Log file to save output')
    args = parser.parse_args()

    setup_logging(args.log_file)

    run_fine_tune_esm(args.csv_file)
