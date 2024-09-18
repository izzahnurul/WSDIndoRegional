import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import random
from huggingface_hub import login

# Set the target language (lan)
lan = "Minangkabau"  # Change to "Sundanese" or "Balinese" as needed

# Login to Hugging Face
hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
login(token='')  # Add your Hugging Face token here

# Load the model and tokenizer for LLaMA 3-8B
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the original dataset and the augmented sentences dataset
datapath = ""  # Replace with preprocessed data path
df = pd.read_excel(datapath)
augmentedpath = ""  # Replace with augmented data path
augmented_df = pd.read_csv(augmentedpath)

# Function to generate the few-shot prompt using augmented sentences
def generate_prompt(df, augmented_df, word):
    # Get augmented examples for few-shot learning
    augmented_examples = augmented_df[augmented_df['target_lan'] == word]

    # Create example sentences from the augmented data
    example_sentences = []
    for _, row in augmented_examples.iterrows():
        example_sentences.append(f"""
        For {lan} word {row['target_lan']} with its translation {row['ind']}, the sentence examples are:
        1. {row['generated_sentence1']}.
        2. {row['generated_sentence2']}.
        3. {row['generated_sentence3']}.
        4. {row['generated_sentence4']}.
        5. {row['generated_sentence5']}.
        """)

    example_text = "\n".join(example_sentences)

    # Get two random examples for disambiguation from the original dataset
    example = df[df['target_lan'] == word].sample(2)
    target_lan_word = word
    target_lan_sentence1 = example.iloc[0]['target_lan_p']
    target_lan_sentence2 = example.iloc[1]['target_lan_p']
    ind_sentence1 = example.iloc[0]['ind_p']
    ind_sentence2 = example.iloc[1]['ind_p']

    target_lan_sentence = random.choice([target_lan_sentence1, target_lan_sentence2])

    # Construct the few-shot prompt
    prompt = f"""{example_text}

Given a word and its sentence in {lan} and two sentences in Indonesian, pick the Indonesian sentence with the correct word sense for the {lan} word:
***
The {lan} word is: {target_lan_word}
The {lan} sentence is: {target_lan_sentence}
The Indonesian sentence A is: {ind_sentence1}
The Indonesian sentence B is: {ind_sentence2}
The correct sense is:
(Please answer with this template: 'Answer: A' or 'Answer: B' and only that)
"""
    return prompt, target_lan_sentence, ind_sentence1, ind_sentence2

# Function to extract the correct answer from the DataFrame
def get_correct_answer(df, target_lan_sentence):
    row = df[df['target_lan_p'] == target_lan_sentence]
    if not row.empty:
        return row.iloc[0]['ind_p']
    return None

# Function to convert the correct answer to A or B
def get_answer_choice(ind_sentence1, ind_sentence2, correct_ind_sentence):
    if correct_ind_sentence == ind_sentence1:
        return "A"
    elif correct_ind_sentence == ind_sentence2:
        return "B"
    return None

# Function to extract the answer from the response
def extract_answer(response):
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("Answer:"):
            return line.split("Answer:")[1].strip()
    return None

# Function to evaluate the response
def evaluate_response(answer, correct_answer):
    return "Correct" if answer == correct_answer else "Incorrect"

# Initialize tracking for total and correct predictions
total_predictions = 0
correct_predictions = 0

# List of unique words
unique_words = df['target_lan'].unique()

word_count = 0

# Loop through each unique word in the dataset
for word in unique_words:
    prompt, target_lan_sentence, ind_sentence1, ind_sentence2 = generate_prompt(df, augmented_df, word)

    # Get the correct answer
    correct_ind_sentence = get_correct_answer(df, target_lan_sentence)
    correct_answer = get_answer_choice(ind_sentence1, ind_sentence2, correct_ind_sentence)
    print(correct_answer)

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the response using the LLaMA 3-8B model
    outputs = model.generate(**inputs, max_length=250, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Prompt:")
    print(prompt)
    print("\nResponse:")
    print(response)

    # Extract the answer from the response
    answer = extract_answer(response)
    print(f"Extracted Answer: {answer}")

    # Evaluate the response
    evaluation = evaluate_response(answer, correct_answer)
    print(f"Evaluation: {evaluation}")
    print("\n" + "="*50 + "\n")

    # Update the prediction counts
    total_predictions += 1
    if evaluation == "Correct":
        correct_predictions += 1

    word_count += 1
    print('count ', word_count)

# Calculate and print the accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
