import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

sai = SentimentIntensityAnalyzer()

Texts = []
Number_of_text = int(input("Enter number of texts: "))

for i in range(1, Number_of_text + 1):
    EText = input(f"Enter text {i}: ")
    Texts.append(EText)

print(Texts)

for i, text in enumerate(Texts, start=1):
    sentiment_score = sai.polarity_scores(text)

    if sentiment_score['compound'] >= 0:
        sentiment = "positive text statement"
    elif sentiment_score['compound'] <= 0:
        sentiment = "negative text statement"
    else:
        sentiment = "neutral text statement"

    print()
    print("Analysis of Text {}: ".format(i))
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print("Sentiment Scores: ", sentiment_score)
    print()
