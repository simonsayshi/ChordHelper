import torch    
from torch.utils.data import Dataset
import pandas as pd
import ast
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChordDataset(Dataset):
    def __init__(self, file_path: str, seq_len: int, pad_token_id: int = 0):
        """
        Args:
            file_path: Path to the tokenized_data_clean.txt
            seq_len: The fixed context window size (e.g., 256)
            pad_token_id: The integer used for padding inputs
        """
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        
        logger.info(f"Loading data from {file_path}...")
        try:
            self.df = pd.read_csv(file_path, on_bad_lines='skip')
            self.raw_chords = self.df.iloc[:, -1].dropna()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

        self.samples = []
        self._process_data()

    def _process_data(self):
        logger.info("Processing chords...")
        valid_sequences = 0
        
        for row in self.raw_chords:
            try:
                # Convert string representation to list
                chord_progression = ast.literal_eval(row)
                
                # Filter out empty or extremely short sequences if necessary
                if len(chord_progression) < 2: 
                    continue
                    
                self.samples.append(chord_progression)
                valid_sequences += 1
                        
            except (ValueError, SyntaxError):
                continue
                
        logger.info(f"Loaded {valid_sequences} sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Get raw sequence
        raw_seq = self.samples[idx]
        
        # Truncate if longer than seq_len + 1 (Input + Target)
        # We need 1 extra token to shift for targets
        max_load_len = self.seq_len + 1
        seq = raw_seq[:max_load_len]
        
        # Convert to tensor
        seq_tensor = torch.tensor(seq, dtype=torch.long)
        
        x = seq_tensor[:-1]
        y = seq_tensor[1:]
        
        # Padding Calculation
        pad_len = self.seq_len - len(x)
        
        if pad_len > 0:
            # Pad input with pad_token_id (0)
            x_pad = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            x = torch.cat([x, x_pad])
            
            # Pad target with -100 (PyTorch Ignore Index)
            # This ensures the model doesn't get penalized for predicting padding
            y_pad = torch.full((pad_len,), -100, dtype=torch.long)
            y = torch.cat([y, y_pad])

        return x, y
