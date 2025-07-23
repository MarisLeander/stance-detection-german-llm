# Description
This is the repo for my bachelors thesis 
- Title: Appeal, Align, Divide? Stance Detection on Group-Directed Messaging in German Political Texts Using Large Language Models
- Submitted to: Data and Web Science Group, Prof. Dr. Simone Paolo Ponzetto, University of Mannheim.
- Supplementary material, as the build database can be found here: https://drive.google.com/drive/folders/1ZuMQNow-ZOQVzNSS2GgwxcOnEz5UNhOq?usp=sharing

# Requirements
- For the local run LLM (gemma-3-27b-it) the minimun requirement is a NVIDIA H100 NVL GPU with 94 GB of VRAM is needed
- A secrets.json needs to be placed in the projects folder to reproduce the output and run the scripts. It has to have the following format:
  {
    "gemini_api_key":"YOUR_API_KEY",
    "huggingface_api_key":"YOUR_API_KEY"
  }
- The paths are coded to be respective to the main dir of your system and the project (stance-detection-german-llm) should thus be placed in your systems home_dir.

# Running the Group Mention Classifications
This is only necesarry, if the mentioned should be re-classified, as the already build db already has the classified mentions.
## Preliminaries
To classify the group mentions, the database has to be build. It his highly recommended to download the pre-build database from [here]{https://drive.google.com/drive/folders/1cO_2MmCOKK2pqSWUgwWIKWYSs3Pux8RG?usp=sharing} to save time. 
Otherwise the database can also build manually, by parsing the downloaded German parliamentary debates. Those can be downloaded [here]{https://drive.google.com/drive/folders/1cO_2MmCOKK2pqSWUgwWIKWYSs3Pux8RG?usp=sharing}. Then one has to run the notbook save_plenary_minutes.ipynb.

# Annotation Data

## Extraction
To extract data for annoation one simple has to run data-processing/extract_annotation_data.py

## Insertion
To insert annotated data one simply has to run data-processing/process_annotated_data.py with the argument --reset_db passed via CLI, to reset the corresponding tables. The Annotators files have to be placed earlier in the /data/annotated_data folder (see main function).

## Calculate Annotator Agreement
The Annotator agreement can be calculated with running data-processing/annotator_agreement.py

## Run the classifier
The bert-base-german-cased-finetuned-MOPE-L3_Run\_3_Epochs_29 classifier has to be downloaded from [here]{https://github.com/umanlp/mope} and placed into a models folder inside the project folder. Then extract-group-mention/extract_group_mention.py has to be run. The --reset_db argument has to be passed via CLI, to reset the corresponding tables.


# Running the Gemini Model
First delete the corresponding data from the build database (DELETE FROM predictions WHERE ...)
## Engineering Dataset
Just run inference/gemini_engineering_inference.py
## Test Dataset
- To allow multi-processing inference/gemini_inference.py has to be called via CLI like this, for each prompt_type / technique combination: python gemini_inference.py --api-key=YOUR_GEMINI_API_KEY --prompt-type=it-thinking_guideline_higher_standards --technique='zero-shot'
- Then the data has to be inserted into the db like this: INSERT INTO predictions
    SELECT paragraph_id, model, prompt_type, technique, prediction, thoughts, thinking_process FROM read_csv('it-thinking_guideline_higher_standards_zero-shot.csv')
    ON CONFLICT DO NOTHING;
- The prompts which can be called like this can be looked up in the script inference/inference_helper.py


# Running the Gemma Model
## Engineering Dataset
Just run inference/gemma_inference.py
## Test Dataset
Just run inference/gemma_inference.py

# Adding new Prompt Types
New prompt types can be added in the inference_helper.py script

# Known issues
The inference helper is a bit messy and it would be advised to rewrite this, to dynamically parse prompts from a json file and adjust it for each prompt_type / technique combination


