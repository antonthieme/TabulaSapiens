import os
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

src_path = Path("../src")
sys.path.append(str(src_path))
src_path = Path("../../src")
sys.path.append(str(src_path))

import ts_tf.custom_esm as cesm
import ts_tf.util as util


def process_pwm(pwm_str: str) -> np.ndarray:
    try:
        list_of_lists = ast.literal_eval(pwm_str)
        return np.array(list_of_lists)
    except Exception as e:
        raise ValueError(f"Error processing PWM: {e}, PWM string: {pwm_str}")

def fine_tune_esm(model_name: str, csv_file: str, save_path: str, epochs: int = 3, lr: float = 2e-5, k_folds: int = 5, batch_size: int = 8):
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
        example_dataset = cesm.ProteinDNADataset(sequences, pwms, padding=(max_rows, max_cols)) # why defined here and then train and validation again
        print(f'example_dataset.max_rows: {example_dataset.max_rows}')
        print(f'example_dataset.max_cols: {example_dataset.max_cols}')
        model = cesm.CustomEsmForPWM(model_name=model_name, output_shape=(example_dataset.max_rows, example_dataset.max_cols))
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

            train_dataset = cesm.ProteinDNADataset(train_sequences, train_pwms, padding=(max_rows, max_cols))
            val_dataset = cesm.ProteinDNADataset(val_sequences, val_pwms, padding=(max_rows, max_cols))
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
        os.makedirs(save_path, exist_ok=True)
        model.save_model(save_path)
        logging.info(f"Model saved to '{save_path}'.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"Empty or invalid CSV file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune ESM for protein-DNA interaction prediction.")
    parser.add_argument('config_file', type=str, help='Config file containing script and model parameters')
    parser.add_argument('job_id', type=str, help='Log file to save output')
    args = parser.parse_args()

    util.setup_logging(f"esm_finetune_{args.job_id}.log")
    config = util.load_config(args.config_file)
    logging.info(f"Loaded configuration: {config}")
    
    try:
        model_save_dir = config['model']['save_dir']
        model_name = config['model']['name']
        epochs = config['training']['epochs']
        lr = config['training']['learning_rate']
        k_folds = config['training']['k_folds']
        batch_size = config['training']['batch_size']
        csv_file = config['data']['csv_file']
        run_dir = config['run']['run_dir']
    except KeyError as e:
        logging.warning(f"Variable not defined in config file. {e}")

    model_save_path = os.path.join(model_save_dir, model_name, args.job_id)
    summary_save_path = os.path.join(run_dir, f'esm_finetune_{args.job_id}')
    writer = SummaryWriter(summary_save_path)

    try:
        fine_tune_esm(model_name=model_name, csv_file=csv_file, save_path=model_save_path, epochs=epochs, lr=lr, k_folds=k_folds, batch_size=batch_size)
    except Exception as e:
        logging.critical(f"Fine-tuning process failed: {e}")
