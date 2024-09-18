from openai import OpenAI
import pandas as pd
import os

# Set the target language (lan)
lan = "Minangkabau"  # Change to "Sundanese" or "Balinese" as needed

# Initialize the OpenAI client
client = OpenAI(
  api_key=""  # Replace with API key
)

# Function to generate the prompt and get the response from GPT
def generate_augmented_sentence(target_lan_word, ind_translation, example_target_lan_sentence, example_ind_sentence):
    prompt = f"""
    Generate ONLY ONE sentence in {lan} for the word '{target_lan_word}' with the Indonesian translation '{ind_translation}', similar to the following pair:
    {lan}: "{example_target_lan_sentence}"
    Indonesian: "{example_ind_sentence}"

    (Please answer with this template: 'Sentence: "(generated sentence)"' and only that)
    """
    # API call to get the response from GPT
    response = client.chat.completions.create(
      model="gpt-4",
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
      ],
      temperature=0.5,
      top_p=1,
      max_tokens=1024
    )

    response_content = response.choices[0].message.content
    generated_sentence = response_content.split('Sentence: ')[-1].strip('"')

    return generated_sentence

# Load existing augmented data if present
if os.path.exists('augmented_data.csv'):
    augmented_df = pd.read_csv('augmented_data.csv')
    processed_indices = set(augmented_df.index)
else:
    augmented_df = pd.DataFrame(columns=['target_lan', 'ind', 'generated_sentence1', 'generated_sentence2', 'generated_sentence3', 'generated_sentence4', 'generated_sentence5'])
    processed_indices = set()

datapath = ""  # Replace with preprocessed data path
df = pd.read_excel(datapath)

# Process each row in the DataFrame and augment the data
augmented_data = []
count = 0

for idx, row in df.iterrows():
    if idx in processed_indices:
        continue

    target_lan_word = row['target_lan']
    ind_translation = row['ind']
    example_target_lan_sentence = row['target_lan_p']
    example_ind_sentence = row['ind_p']

    generated_sentences = []
    attempts = 0

    while len(generated_sentences) < 5 and attempts < 10:  # Allow up to 10 attempts to generate 5 unique sentences
        generated_sentence = generate_augmented_sentence(target_lan_word, ind_translation, example_target_lan_sentence, example_ind_sentence)
        if generated_sentence not in generated_sentences:
            generated_sentences.append(generated_sentence)
        attempts += 1

    # Append the original and augmented data
    augmented_data.append((target_lan_word, ind_translation, *generated_sentences))

    # Print the progress
    print(count)
    print(f"Original: {example_target_lan_sentence} - {example_ind_sentence}")
    for i, sentence in enumerate(generated_sentences, 1):
        print(f"Generated {i}: {sentence}")
    print("="*50)

    count += 1

    # Save intermediate results every 10 iterations
    if count % 10 == 0:
        temp_df = pd.DataFrame(augmented_data, columns=['target_lan', 'ind', 'generated_sentence1', 'generated_sentence2', 'generated_sentence3', 'generated_sentence4', 'generated_sentence5'])
        augmented_df = pd.concat([augmented_df, temp_df])
        augmented_df.to_csv('augmented_data.csv', index=False)
        augmented_data = []

# Save final results
if augmented_data:
    temp_df = pd.DataFrame(augmented_data, columns=['target_lan', 'ind', 'generated_sentence1', 'generated_sentence2', 'generated_sentence3', 'generated_sentence4', 'generated_sentence5'])
    augmented_df = pd.concat([augmented_df, temp_df])
    augmented_df.to_csv('augmented_data.csv', index=False)

print("Augmented data has been generated.")
