from pathlib import Path
from functools import partial
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorWithPadding,
)

# --- WORKER FUNCTION (Top Level) ---
def tokenize_batch_worker(batch, tokenizer):
    """Tokenizes a batch of text and attaches word_ids."""
    tokenized_inputs = tokenizer(batch["paragraphs"], truncation=True, is_split_into_words=False)
    word_ids_list = [tokenized_inputs.word_ids(i) for i in range(len(tokenized_inputs["input_ids"]))]
    tokenized_inputs["word_ids"] = word_ids_list
    return tokenized_inputs

# --- SOLUTION: Moved the collate function to the top level ---
def custom_collate_fn_top_level(features, tokenizer):
    """
    Custom collate function that handles word_ids separately to prevent
    them from being converted to tensors.
    """
    # Keep word_ids as a standard list of lists
    word_ids_list = [feature.pop("word_ids") for feature in features]
    
    # Use the default collator for the rest of the features (input_ids, etc.)
    batch = tokenizer.pad(
        features,
        padding=True,
        return_tensors="pt",
    )
    
    # Add the non-tensor word_ids list back into the final batch
    batch["word_ids"] = word_ids_list
    return batch

def index2label(index):
    """ Converts a label index back to its string representation. """
    labels = {0: '[PAD]', 1: '[UNK]', 2: 'B-EGPOL', 3: 'B-EOFINANZ', 4: 'B-EOMEDIA', 5: 'B-EOMIL', 6: 'B-EOMOV', 7: 'B-EONGO', 8: 'B-EOPOL', 9: 'B-EOREL', 10: 'B-EOSCI', 11: 'B-EOWIRT', 12: 'B-EPFINANZ', 13: 'B-EPKULT', 14: 'B-EPMEDIA', 15: 'B-EPMIL', 16: 'B-EPMOV', 17: 'B-EPNGO', 18: 'B-EPPOL', 19: 'B-EPREL', 20: 'B-EPSCI', 21: 'B-EPWIRT', 22: 'B-GPE', 23: 'B-PAGE', 24: 'B-PETH', 25: 'B-PFUNK', 26: 'B-PGEN', 27: 'B-PNAT', 28: 'B-PSOZ', 29: 'I-EGPOL', 30: 'I-EOFINANZ', 31: 'I-EOMEDIA', 32: 'I-EOMIL', 33: 'I-EOMOV', 34: 'I-EONGO', 35: 'I-EOPOL', 36: 'I-EOREL', 37: 'I-EOSCI', 38: 'I-EOWIRT', 39: 'I-EPFINANZ', 40: 'I-EPKULT', 41: 'I-EPMEDIA', 42: 'I-EPMIL', 43: 'I-EPMOV', 44: 'I-EPNGO', 45: 'I-EPPOL', 46: 'I-EPREL', 47: 'I-EPSCI', 48: 'I-EPWIRT', 49: 'I-GPE', 50: 'I-PAGE', 51: 'I-PETH', 52: 'I-PFUNK', 53: 'I-PGEN', 54: 'I-PNAT', 55: 'I-PSOZ', 56: 'O'}
    return labels.get(index, "[UNK]")


class GroupClassifier:
    def __init__(self, model_dir, device=None):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        base_model_name = "bert-base-german-cased"
        print(f"Loading FAST tokenizer from base model: '{base_model_name}'")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        
        print(f"Loading fine-tuned model from: '{self.model_dir}'")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16
        ).to(self.device)

        if torch.__version__ >= "2.0.0":
            print("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)
        
        self.model.eval()
        print("Model loaded and ready.")

    def _prepare_data(self, data: list[dict], batch_size: int, num_workers: int):
        ds = Dataset.from_list(data)
        
        tokenize_func = partial(tokenize_batch_worker, tokenizer=self.tokenizer)

        encoded_data = ds.map(
            tokenize_func,
            batched=True,
            remove_columns=["paragraphs"],
            num_proc=num_workers
        )
        
        # --- SOLUTION: Use partial to create the collate function ---
        collate_fn = partial(custom_collate_fn_top_level, tokenizer=self.tokenizer)

        return DataLoader(
            encoded_data,
            batch_size=batch_size,
            collate_fn=collate_fn, # Use the top-level collate function
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=str(self.device) if self.device.type == 'cuda' else ''
        )

    def predict(self, paragraphs: list[str], batch_size: int = 64, num_workers: int = 4):
        if not paragraphs:
            return []
            
        data_dicts = [{"paragraphs": p} for p in paragraphs]
        dataloader = self._prepare_data(data_dicts, batch_size, num_workers)
        
        all_predictions = []
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
            for batch in dataloader:
                word_ids_batch = batch.pop("word_ids")
                
                batch_on_device = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                output = self.model(**batch_on_device)
                
                predictions = torch.argmax(output.logits, dim=-1).cpu()

                for i in range(len(predictions)):
                    word_ids = word_ids_batch[i]
                    current_preds = predictions[i]
                    current_tokens = self.tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
                    
                    aligned_pairs = []
                    previous_word_idx = None
                    for token_idx, word_idx in enumerate(word_ids):
                        if word_idx is None or word_idx == previous_word_idx:
                            continue
                        
                        previous_word_idx = word_idx
                        # Get all subword tokens for the current word
                        subword_indices = [j for j, w_id in enumerate(word_ids) if w_id == word_idx]
                        
                        # Use the label from the first subword for the entire word
                        label = index2label(current_preds[subword_indices[0]].item())
                        
                        # --- BUG FIX: Reconstruct the word from its tokens, not from the original string ---
                        subword_tokens = [current_tokens[k] for k in subword_indices]
                        full_word = self.tokenizer.convert_tokens_to_string(subword_tokens)

                        aligned_pairs.append((full_word, label))

                    all_predictions.append(aligned_pairs)
        
        return all_predictions
