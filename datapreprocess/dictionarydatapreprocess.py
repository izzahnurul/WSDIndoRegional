import pandas as pd
import warnings
warnings.filterwarnings('ignore')

entries_path = ""  # path to the dictionary entries here
phrases_path = ""  # path to the dictionary phrases here

# Load the entries and phrases datasets
entries_df = pd.read_excel(entries_path)
phrases_df = pd.read_excel(phrases_path)

# Set the target language (lan)
lan = "Minangkabau"  # Change this to "Balinese" or "Sundanese" as needed

# Function to adjust column names based on the language
def adjust_columns_for_language(lan, df, is_phrase=False):
    if lan == "Balinese":
        target_col = 'bal'
        target_col_p = 'bal_p'
    elif lan == "Minangkabau":
        target_col = 'min'
        target_col_p = 'min_p'
    elif lan == "Sundanese":
        target_col = 'sun'
        target_col_p = 'sun_p'
    else:
        raise ValueError("Unsupported language")

    if is_phrase:
        return df.rename(columns={target_col_p: 'target_lan_p'})
    else:
        return df.rename(columns={target_col: 'target_lan'})

# Adjust the columns of the entries and phrases DataFrames
entries_df = adjust_columns_for_language(lan, entries_df)
phrases_df = adjust_columns_for_language(lan, phrases_df, is_phrase=True)

# Filter words with multiple meanings
multiple_meanings = entries_df.groupby('target_lan').filter(lambda x: x['ind'].nunique() > 1)

results = []

# Iterate over the unique words in the target language
for word in multiple_meanings['target_lan'].unique():
    meanings = entries_df[entries_df['target_lan'] == word]
    phrases = phrases_df[phrases_df['target_lan_p'].str.contains(word)]

    for _, meaning_row in meanings.iterrows():
        for _, phrase_row in phrases.iterrows():
            if meaning_row['ind'] in phrase_row['ind_p']:
                results.append([word, meaning_row['ind'], phrase_row['target_lan_p'], phrase_row['ind_p']])

# Create a DataFrame with the results
df = pd.DataFrame(results, columns=['target_lan', 'ind', 'target_lan_p', 'ind_p'])

# Identify words with multiple meanings
multiple_meanings = df.groupby('target_lan').filter(lambda x: x['ind'].nunique() > 1)

# Store selected rows
selected_rows = []

# Iterate over the unique words with multiple meanings
for word in multiple_meanings['target_lan'].unique():
    meanings = multiple_meanings[multiple_meanings['target_lan'] == word]
    for meaning in meanings['ind'].unique():
        example = meanings[meanings['ind'] == meaning].iloc[0]
        selected_rows.append(example)

# Create the final DataFrame
final_result = pd.DataFrame(selected_rows)

# Display the final result
print(final_result)
