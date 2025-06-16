#%%
from pathlib import Path

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorWithPadding

import torch
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
import numpy as np


def index2label(index):
    """ Convert an index to a label.

    Args:
        index (int): The index to be converted.

    Returns:
        str: The label corresponding
    """

    labels = {0: '[PAD]', 1: '[UNK]', 2: 'B-EGPOL', 3: 'B-EOFINANZ', 4: 'B-EOMEDIA', 5: 'B-EOMIL', 6: 'B-EOMOV', 7: 'B-EONGO', 8: 'B-EOPOL', 9: 'B-EOREL', 10: 'B-EOSCI', 11: 'B-EOWIRT', 12: 'B-EPFINANZ', 13: 'B-EPKULT', 14: 'B-EPMEDIA', 15: 'B-EPMIL', 16: 'B-EPMOV', 17: 'B-EPNGO', 18: 'B-EPPOL', 19: 'B-EPREL', 20: 'B-EPSCI', 21: 'B-EPWIRT', 22: 'B-GPE', 23: 'B-PAGE', 24: 'B-PETH', 25: 'B-PFUNK', 26: 'B-PGEN', 27: 'B-PNAT', 28: 'B-PSOZ', 29: 'I-EGPOL', 30: 'I-EOFINANZ', 31: 'I-EOMEDIA', 32: 'I-EOMIL', 33: 'I-EOMOV', 34: 'I-EONGO', 35: 'I-EOPOL', 36: 'I-EOREL', 37: 'I-EOSCI', 38: 'I-EOWIRT', 39: 'I-EPFINANZ', 40: 'I-EPKULT', 41: 'I-EPMEDIA', 42: 'I-EPMIL', 43: 'I-EPMOV', 44: 'I-EPNGO', 45: 'I-EPPOL', 46: 'I-EPREL', 47: 'I-EPSCI', 48: 'I-EPWIRT', 49: 'I-GPE', 50: 'I-PAGE', 51: 'I-PETH', 52: 'I-PFUNK', 53: 'I-PGEN', 54: 'I-PNAT', 55: 'I-PSOZ', 56: 'O'}

    return labels[index]
    #
    # labels = ["[PAD]", "[UNK]", "B-EGPOL", "B-EOFINANZ", "B-EOMEDIA", "B-EOMIL", "B-EOMOV", "B-EONGO", "B-EOPOL", "B-EOREL", "B-EOSCI", "B-EOWIRT", "B-EPFINANZ", "B-EPKULT", "B-EPMEDIA", "B-EPMIL", "B-EPMOV", "B-EPNGO", "B-EPPOL", "B-EPREL", "B-EPSCI", "B-EPWIRT", "B-GPE", "B-PAGE", "B-PETH", "B-PFUNK", "B-PGEN", "B-PNAT", "B-PSOZ", "I-EGPOL", "I-EOFINANZ", "I-EOMEDIA", "I-EOMIL", "I-EOMOV", "I-EONGO", "I-EOPOL", "I-EOREL", "I-EOSCI", "I-EOWIRT", "I-EPFINANZ", "I-EPKULT", "I-EPMEDIA", "I-EPMIL", "I-EPMOV", "I-EPNGO", "I-EPPOL", "I-EPREL", "I-EPSCI", "I-EPWIRT", "I-GPE", "I-PAGE", "I-PETH", "I-PFUNK", "I-PGEN", "I-PNAT", "I-PSOZ", "O"]
    # label2index, index2label = {}, {}
    # for i, item in enumerate(labels):
    #     label2index[item] = i
    #     index2label[i] = item


class GroupClassifier:
    def __init__(self, model_dir, device=None):
        """
        Initializes the classifier by loading the model and tokenizer once.
        This is the slow part, and it now only runs when a TokenClassifier object is created.
        """
        self.model_dir = Path(model_dir)
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {self.model_dir} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        
        # Using float16 for A100 performance
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 
        ).to(self.device)

        # Optional: Compile the model for a potential speed boost on PyTorch 2.0+
        # The first prediction will be slower due to a one-time compilation cost.
        # self.model = torch.compile(self.model, mode="max-autotune")
        
        self.model.eval()
        print("Model loaded and ready.")

    def _prepare_data(self, data: list[dict[str, list[str]]], batch_size: int):
        ds = Dataset.from_list(data)
        
        def tokenize_batch(batch):
            return self.tokenizer(batch["paragraphs"], truncation=True, is_split_into_words=False)

        encoded_data = ds.map(tokenize_batch, batched=True, remove_columns=["paragraphs"])
        encoded_data.set_format("torch")
        
        return DataLoader(encoded_data, batch_size=batch_size, collate_fn=DataCollatorWithPadding(self.tokenizer))

    def predict(self, paragraphs: list[str], batch_size: int = 64):
        """
        Runs predictions on a list of paragraphs. This method is fast.
        
        Args:
            paragraphs (list[str]): A list of sentences/paragraphs to classify.
            batch_size (int): The number of samples to process at once. Tune for your GPU.
        
        Returns:
            list[list[tuple[str, str]]]: A list where each inner list contains (token, label) pairs for a paragraph.
        """
        if not paragraphs:
            return []
            
        data_dicts = [{"paragraphs": p} for p in paragraphs]
        dataloader = self._prepare_data(data_dicts, batch_size)
        
        all_predictions = []
        with torch.inference_mode(): # Ensures no gradients are calculated
            with torch.autocast(device_type=self.device.type, dtype=torch.float16): # Enables mixed-precision
                for batch in dataloader:
                    batch_on_device = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    output = self.model(**batch_on_device)
                    
                    logits = output.logits
                    predictions = torch.argmax(logits, dim=-1).cpu()
                    input_ids_cpu = batch_on_device["input_ids"].cpu()
                    
                    for i in range(len(predictions)):
                        # Move the attention mask for the current item to the CPU before creating the boolean mask.
                        attention_mask_cpu = batch_on_device["attention_mask"][i].cpu()
                        
                        # Now use the CPU mask to index the CPU tensors.
                        valid_token_ids = input_ids_cpu[i][attention_mask_cpu == 1]
                        valid_preds = predictions[i][attention_mask_cpu == 1]

                        tokens = self.tokenizer.convert_ids_to_tokens(valid_token_ids, skip_special_tokens=True)
                        labels = [index2label(p.item()) for p in valid_preds[1:-1]] # Exclude [CLS] and [SEP] predictions
                        
                        # Simple alignment, might need improvement for complex cases
                        aligned_labels = labels[:len(tokens)] 
                        
                        all_predictions.append(list(zip(tokens, aligned_labels)))
        
        return all_predictions


#%%

# def merge_word_pieces(paired_tokens):
#     """
#     paired_tokens : List[Tuple[str, str]]
#         e.g. [("Beispiel","O"), ("##satz","O"), ...]
#
#     Returns
#     -------
#     merged : List[Tuple[str, str]]
#         Word-level (token, label) pairs.
#     """
#     merged = []
#     current_word = ""
#     current_label = None
#
#     # Maybe adjust this set to tokenizer's special tokens
#     SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
#
#     for token, label in paired_tokens:
#         # 1. drop specials outright
#         if token in SPECIAL:
#             continue
#
#         # 2. continuation piece?
#         if token.startswith("##"):
#             current_word += token[2:]        # append stem
#             continue                         # label already set
#         else:
#             # 3. flush previous buffered word
#             if current_word:
#                 merged.append((current_word, current_label))
#             # 4. start new word buffer
#             current_word  = token
#             current_label = label
#
#     # flush last word
#     if current_word:
#         merged.append((current_word, current_label))
#     return merged