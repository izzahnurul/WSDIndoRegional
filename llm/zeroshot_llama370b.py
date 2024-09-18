import pandas as pd
from openai import OpenAI
import random

# Set the target language (lan)
lan = "Minangkabau"  # Change to "Sundanese" or "Balinese" as needed

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=""  # Replace with your API key
)

datapath = ""  # Replace with the preprocessed data path
df = pd.read_excel(datapath)

# Function to generate the prompt
def generate_prompt(df, word):
    example = df[df['target_lan'] == word].sample(2)
    target_lan_word = word
    target_lan_sentence = random.choice([example.iloc[0]['target_lan_p'], example.iloc[1]['target_lan_p']])
    ind_sentence1 = example.iloc[0]['ind_p']
    ind_sentence2 = example.iloc[1]['ind_p']

    prompt = f"""Given a word and its sentence in {lan} and two sentences in Indonesian, pick the Indonesian sentence with the correct word sense for the {lan} word:
***
The {lan} word is: {target_lan_word}
The {lan} sentence is: {target_lan_sentence}
The Indonesian sentence A is: {ind_sentence1}
The Indonesian sentence B is: {ind_sentence2}
The correct sense is:
(Answer with this template: 'Answer: A' or 'Answer: B')
"""
    return prompt, target_lan_sentence, ind_sentence1, ind_sentence2

# Function to get the correct answer from the DataFrame
def get_correct_answer(df, target_lan_sentence):
    row = df[df['target_lan_p'] == target_lan_sentence]
    return row.iloc[0]['ind_p'] if not row.empty else None

# Convert correct answer to A or B
def get_answer_choice(ind_sentence1, ind_sentence2, correct_ind_sentence):
    return "A" if correct_ind_sentence == ind_sentence1 else "B" if correct_ind_sentence == ind_sentence2 else None

# Extract the answer from the response
def extract_answer(response):
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("Answer:"):
            return line.split("Answer:")[1].strip()
    return None

# Evaluate the response
def evaluate_response(answer, correct_answer):
    return "Correct" if answer == correct_answer else "Incorrect"

# Variables to track the total and correct predictions
total_predictions = 0
correct_predictions = 0

# Main loop
word_count = 0
unique_words = df['target_lan'].unique()

for word in unique_words:
    prompt, target_lan_sentence, ind_sentence1, ind_sentence2 = generate_prompt(df, word)
    correct_ind_sentence = get_correct_answer(df, target_lan_sentence)
    correct_answer = get_answer_choice(ind_sentence1, ind_sentence2, correct_ind_sentence)

    # API call to get the response from LLaMA 3 70B
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024
    )

    response = completion.choices[0].message.content
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
    print('Processed word count: ', word_count)

# Calculate and print accuracy
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy * 100:.2f}%")
