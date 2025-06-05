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
#%%
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

def load_model(model_dir):
    """ Load a pre-trained model from the specified directory.

    Args:
        model_dir (str or Path): The directory where the model is stored.

    Returns:
        model (AutoModelForTokenClassification): The loaded model.
    """
    # Load the config
    cfg   = AutoConfig.from_pretrained(model_dir)
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_dir, config=cfg).to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def tokenize_labels(data, tokenizer):
    """ Tokenize the input data. This is needed to prepare the input data for the model.

    Args:
        data (dict): The input data containing the paragraphs to be tokenized.
        tokenizer (AutoTokenizer): The tokenizer to be used for tokenization.

    Returns:
        tokenized_inputs (dict): A dictionary containing the tokenized inputs.
    """
    tokenized_inputs = tokenizer(data["paragraphs"],
                                  truncation=True,
                                  padding=True,
                                  is_split_into_words=False)

    return tokenized_inputs

def encode_dataset(raw_data: list[dict[str, list[str]]], tokenizer):
    """ Encode the dataset using the tokenizer. This is needed to prepare the input data for the model.
    Args:
        raw_data (list[dict[str, list[str]]]): The raw data to be encoded. Each entry in the list corresponds to a paragraph from a speech.
        tokenizer: The tokenizer to be used for encoding.
    Returns:
        dataset (Dataset): The encoded dataset.
    """

    # data = load_dataset('json', data_files={'input_data': path}) -> Use this to import from json
    ds = Dataset.from_list(raw_data)
    data = DatasetDict({"input_data": ds})
    encoded_data = data.map(
                tokenize_labels,              # ← function reference
                batched=True,
                fn_kwargs={"tokenizer": tokenizer},  # extra objects you need
                remove_columns=["paragraphs"]# only keep the columns that are needed, i.e. input_ids, attention_mask, token_type_ids and labels. It checks if the columns are present in the corpus and removes them if they are not needed.
            )
    dataset = encoded_data["input_data"].with_format("torch")  # Convert to PyTorch format, to be compatible with DataLoader
    return dataset

def build_dataloader(dataset, tokenizer, batch_size=16):
    """ Build a DataLoader for the dataset. The DataLoader will be used to load the data in batches for inference.
    This is beneficial for large datasets, as it allows us to load the data in smaller chunks, which can be processed in parallel.

    Args:
        dataset (Dataset): The dataset to be loaded.
        tokenizer (AutoTokenizer): The tokenizer to be used for tokenization.
        batch_size (int, optional): The batch size to be used. Defaults to 16.
    Returns:
        DataLoader: The DataLoader for the dataset.
    """
    # Create a DataLoader for the test dataset. The DataCollatorWithPadding will pad the sequences to the one with the maximum length in the batch, so that all sequences in the batch have the same length.
    # The dataloader will inherit batches of data, which we can then pass to the model for prediction.
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer))


def get_predictions(dataloader:DataLoader, model, tokenizer):
    """ Get predictions from the model for the given dataloader. This function will iterate over the dataloader and get the predictions for each batch.
    Args:
        dataloader (DataLoader): The DataLoader containing the data to be predicted.
        model (AutoModelForTokenClassification): The model to be used for prediction.
        tokenizer (AutoTokenizer): The tokenizer to be used for tokenization.
    Returns:
        A list of lists containing the predicted labels for each token in each batch.
    """

    preds_all= [] # Initialize lists to store predictions

    for batch in dataloader:
        print("Batch size:", len(batch["input_ids"]))
        with torch.inference_mode():
            output = model(**batch) # unpacks the batch dictionary and passes the input IDs and attention mask to the model, which will return the logits, which are the unnormalized scores for each label.

        logits = output.logits                    # Logits are the unnormalized scores for each label.
        preds  = torch.argmax(logits, dim=-1)     # take the best label for each token. We use argmax for inference since we don't need to compute the loss during inference, we just want the predicted labels. Also we won't do majority voting since (now) we only use one model.
        for i in range(len(preds)):
            labels = [index2label(int(i)) for i in preds[i]]
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
            # preds_all.append(merge_word_pieces(zip(tokens,labels)))
            preds_all.append(list(zip(tokens, labels)))  # Extend the list with the new predictions
    return preds_all
    # maybe store the predictions in a list and compare them with the ground truth labels!!!

def predict_batch(data: list[dict[str, list[str]]]) -> list[list[tuple[str, str]]]:
    """ Predict the labels for a batch of paragraphs. This function will load the model, encode the dataset, build the dataloader and get the predictions.
    Args:
        data (list[dict[str, list[str]]]): The input data containing the paragraphs to be predicted.
    Returns:
        list[list[tuple[str, str]]]: A list of lists containing the predicted labels for each token in each paragraph.
    """

    model_dir = Path("../models/bert-base-german-cased-finetuned-MOPE-L3_Run_2_Epochs_29")
    tokenizer   = AutoTokenizer.from_pretrained(model_dir, use_fast=True) # use_fast=True enables the fast tokenizer implementation
    model = load_model(model_dir)
    encoded_dataset = encode_dataset(data, tokenizer)
    dataloader = build_dataloader(encoded_dataset, tokenizer)
    return get_predictions(dataloader, model, tokenizer)


# if __name__ == "__main__":
#     main()
#%%
# paragraphs = [{'paragraphs': 'Meine Damen und Herren, die Beantwortung der Interpellation ist erfolgt. Ich frage, ob eine Besprechung der Interpellation gewünscht wird. — Das ist nicht der Fall. Damit ist Punkt 1 der Tagesordnung erledigt. Ich rufe auf Punkt 2 der Tagesordnung:'}, {'paragraphs': 'Erste Beratung des Entwurfs eines Gesetzes zur Regelung der Besteuerung des Kleinpflanzertabaks im Erntejahr 1950 (Nr. 1508 der Drucksachen).'}, {'paragraphs': 'Dazu hat zunächst das Wort Herr Staatssekretär Hartmann.'}]
#
# classified_paragraphs = predict_batch(paragraphs)
# print(classified_paragraphs)
#%%

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