# Description
This is the repo for my bachelors thesis "Appeal, Align, Divide? Stance Detection on Group-Directed Messaging in German Political Texts Using Large Language Models", Submitted to: Data and Web Science Group, Prof. Dr. Simone Paolo Ponzetto, University of Mannheim.
Supplementary material, as the build database can be found here: https://drive.google.com/drive/folders/1ZuMQNow-ZOQVzNSS2GgwxcOnEz5UNhOq?usp=sharing

# Requirements
- For the local run LLM (gemma-3-27b-it) the minimun requirement is a NVIDIA H100 NVL GPU with 94 GB of VRAM is needed
- A secrets.json needs to be placed in the projects folder to reproduce the output and run the scripts. It has to have the following format:
  {
    "gemini_api_key":"YOUR_API_KEY",
    "huggingface_api_key":"YOUR_API_KEY"
  }

