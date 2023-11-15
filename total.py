import pandas as pd
import re
import seaborn as sns
import matplotlib as plt

# đếm ký tự đặc biệt
def count_special_characters(text):
    special_characters = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    total_characters = len(text.replace(" ", ""))  
    return special_characters, total_characters

# đếm từ
def count_words(text):
    words = text.split()
    total_words = len(words)
    return total_words

# đếm stop_words
def count_stop_words(text):
    stop_words = set(['a', 'an', 'the', 'in', 'on', 'at', 'for', 'to', 'of', 'is', 'are', 'was', 'were'])  
    words = text.split()
    stop_word_count = len([word for word in words if word.lower() in stop_words])
    return stop_word_count

# đếm giới từ đại từ viết tắt
def count_prepositions_pronouns_abbreviations(text):
    prepositions = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond', 'by', 'down', 'during', 'except', 'for', 'from', 'in', 'inside', 'into', 'like', 'near', 'of', 'off', 'on', 'out', 'outside', 'over', 'past', 'since', 'through', 'to', 'toward', 'under', 'underneath', 'until', 'up', 'upon', 'with', 'within', 'without']
    pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'you', 'him', 'her', 'us', 'them']
    abbreviations = ['NASA', 'FBI', 'UN']

    preposition_count = len([word for word in text.split() if word.lower() in prepositions])
    pronoun_count = len([word for word in text.split() if word.lower() in pronouns])
    abbreviation_count = len([word for word in text.split() if word.upper() in abbreviations])

    return preposition_count, pronoun_count, abbreviation_count

df = pd.read_csv('D:\\fake news\\fake-news\\train.csv')

data = []

for index, row in df.iterrows():
    text = row['text']
    label = row['label'] 

    if isinstance(text, str):
        special_character_count, total_character_count = count_special_characters(text)
        word_count = count_words(text)
        stop_word_count = count_stop_words(text)
        preposition_count, pronoun_count, abbreviation_count = count_prepositions_pronouns_abbreviations(text)

        percentage_special_characters = (special_character_count / total_character_count) * 100 if total_character_count != 0 else 0
        percentage_stop_words = (stop_word_count / word_count) * 100 if word_count != 0 else 0
        percentage_prepositions = (preposition_count / word_count) * 100 if word_count != 0 else 0
        percentage_pronouns = (pronoun_count / word_count) * 100 if word_count != 0 else 0
        percentage_abbreviations = (abbreviation_count / word_count) * 100 if word_count != 0 else 0

        data.append({
            'Text': text,
            'Ký tự đặc biệt': special_character_count,
            '% ký tự đặc biệt': percentage_special_characters,
            'Stop Words': stop_word_count,
            '% Stop Words': percentage_stop_words,
            'giới từ': preposition_count,
            '% giới từ': percentage_prepositions,
            'Đại từ': pronoun_count,
            '% đại từ': percentage_pronouns,
            'Các từ viết tắt': abbreviation_count,
            '% các từ viết tắt': percentage_abbreviations
            
        })

df_result = pd.DataFrame(data)


df_result.to_csv('D:\\fake news\\fake-news\\data1.csv', index=False)

print(df_result)

sns.countplot(x="label", data=df)
print("0: real")
print("1: fake")
print(round(df.label.value_counts(normalize=True), 2) * 100)


df1 = df_result.copy()
df1=df1.drop([ 'Text'], axis=1)
# d1f=df1.drop([ 'Label'], axis=1)
df1.shape
df1.head
# df1.to_csv('D:\\fake news\\fake-news\\data1.csv', index=False)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
# import pandas as pd


df.drop_duplicates(subset='text', keep='first', inplace=True)
df['text'] = df['text'].astype('str')

# Vader sentiment analysis
sid = SentimentIntensityAnalyzer()
df['vader_scores'] = df['text'].apply(lambda text: sid.polarity_scores(text))
df['vader_positive'] = df['vader_scores'].apply(lambda x: x['pos'])
df['vader_negative'] = df['vader_scores'].apply(lambda x: x['neg'])
df['vader_neutral'] = df['vader_scores'].apply(lambda x: x['neu'])

# TextBlob sentiment analysis
df['textblob_scores'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['textblob_positive'] = df['textblob_scores'].apply(lambda x: x if x > 0 else 0)
df['textblob_negative'] = df['textblob_scores'].apply(lambda x: abs(x) if x < 0 else 0)
df['textblob_neutral'] = df['textblob_scores'].apply(lambda x: 1 - abs(x) if abs(x) <= 0.5 else 0)

# Save the updated DataFrame to a CSV file
df[['vader_positive', 'vader_negative', 'vader_neutral', 'textblob_positive', 'textblob_negative', 'textblob_neutral']].to_csv('D:\\fake news\\fake-news\\data1.csv', index=False)
df=df.drop([ 'text'], axis=1)
df=df.drop([ 'title'], axis=1)
df=df.drop([ 'id'], axis=1)
df=df.drop([ 'author'], axis=1)

# df=df.drop([ 'label'], axis=1)
df=df.drop([ 'vader_scores'], axis=1)
df=df.drop([ 'textblob_scores'], axis=1)
df2 = pd.read_csv('D:\\fake news\\fake-news\\data1.csv')
# df2 = pd.concat([df2, df, df1], axis=1)
df2 = pd.concat([df1, df], axis=1)

label_column = df2.pop("label")
df2["Label"] = label_column

df2_filled = df2.fillna(0)
print(df2_filled)
df2_filled.to_csv('D:\\fake news\\fake-news\\data4.csv')
