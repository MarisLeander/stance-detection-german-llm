from transformers import pipeline
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import time

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

start_time = time.time()

# Tell vLLM to use 2 GPUs for tensor parallelism.
# This automatically splits the model across the 2 A100s.
# The process takes about 30min!
llm = llm = LLM(model=model_name, tensor_parallel_size=2)

elapsed_time = (time.time() - start_time) / 60
print(f"LLM setup took {elapsed_time} min.")

# Load the tokenizer for the same model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the tokenizer for the same model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Define the conversation roles
messages = [
    {
        "role": "system",
        "content": "You are a friendly and helpful AI chatbot. Your goal is to assist the user with their questions in a clear and concise way."
    },
    {
        "role": "user",
        "content": "Bitte gib die Speaker stance für den folgenden Text aus den Labeln [Favour, Against, Neither] an. Target: \"der luxemburgischen Ratspräsidentschaft in\" (markiert durch <span> tag). Text: Die Einigung ist für uns deshalb mindestens so entscheidend wie für die anderen europäischen Staaten. Im Juni haben wir die Einigung unter <span>der luxemburgischen Ratspräsidentschaft in</span> Luxemburg schon einmal versucht. Ich sage voraus: Wenn wir am Ende dieses Jahres mit dem zweiten Versuch einer Einigung über den Finanzrahmen erneut scheitern würden, dann ginge davon ein verheerendes Signal für die Bürgerinnen und Bürger aus. Insbesondere darf nicht vergessen werden, dass sich ein Scheitern vor allem zulasten der neuen Mitgliedstaaten auswirken würde."
    }
]

# Applying the chat template to create the final prompt string.
#    - `tokenize=False` makes it return a string, which is what llm.generate expects.
#    - `add_generation_prompt=True` adds the special tokens to signal to the model that it's the assistant's turn to speak.
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# Single formatted prompt for model
prompts_to_generate = [formatted_prompt]


# sampling parameters for generation.
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=1000 # Adjust max_tokens as needed
)

# Generate completions for all prompts in a batch using vllm

outputs = llm.generate(prompts_to_generate, sampling_params)

for output in outputs:
    # The 'prompt' here will be the long, formatted string with special tokens
    original_prompt_info = output.prompt
    generated_text = output.outputs[0].text
    
    print("--- Input to Model ---")
    print(f"{original_prompt_info!r}")
    print("\n--- Generated Output ---")
    print(f"{generated_text!r}")