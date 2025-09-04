import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu labeled
df_labeled = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\labeled_data.csv")

# Vector hóa text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df_labeled["sentence"])

# Train mô hình cho topic
y_topic = df_labeled["topic"]
clf_topic = LogisticRegression(max_iter=1000)
clf_topic.fit(X, y_topic)

# Train mô hình cho sentiment
y_sentiment = df_labeled["sentiment"]
clf_sentiment = LogisticRegression(max_iter=1000)
clf_sentiment.fit(X, y_sentiment)

print("✅ Đã train xong mô hình cơ bản cho Topic và Sentiment")
