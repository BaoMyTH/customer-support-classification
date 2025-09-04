import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === BƯỚC 1: Đọc dữ liệu ===
df_labeled = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\labeled_data.csv")
df_unlabeled = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\unlabeled_data.csv")

# === BƯỚC 2: Vector hóa text ===
vectorizer = TfidfVectorizer(max_features=5000)
X_labeled = vectorizer.fit_transform(df_labeled["sentence"])
X_unlabeled = vectorizer.transform(df_unlabeled["sentence"])

# === BƯỚC 3: Train mô hình cơ bản ===
clf_topic = LogisticRegression(max_iter=1000)
clf_topic.fit(X_labeled, df_labeled["topic"])

clf_sentiment = LogisticRegression(max_iter=1000)
clf_sentiment.fit(X_labeled, df_labeled["sentiment"])

# === BƯỚC 4: Self-training cho Topic ===
proba_topic = clf_topic.predict_proba(X_unlabeled)
high_conf_idx = proba_topic.max(axis=1) > 0.8   # ngưỡng tin cậy thấp hơn

if high_conf_idx.any():
    pseudo_labels_topic = clf_topic.predict(X_unlabeled[high_conf_idx])
    df_pseudo_topic = df_unlabeled[high_conf_idx].copy()
    df_pseudo_topic["topic"] = pseudo_labels_topic
else:
    df_pseudo_topic = pd.DataFrame(columns=["sentence", "topic"])

# === BƯỚC 5: Self-training cho Sentiment ===
proba_sent = clf_sentiment.predict_proba(X_unlabeled)
high_conf_idx_s = proba_sent.max(axis=1) > 0.8

if high_conf_idx_s.any():
    pseudo_labels_sent = clf_sentiment.predict(X_unlabeled[high_conf_idx_s])
    df_pseudo_sent = df_unlabeled[high_conf_idx_s].copy()
    df_pseudo_sent["sentiment"] = pseudo_labels_sent
else:
    df_pseudo_sent = pd.DataFrame(columns=["sentence", "sentiment"])

# === BƯỚC 6: Gộp lại với tập labeled ban đầu ===
df_new_topic = pd.concat([
    df_labeled[["sentence", "topic"]],
    df_pseudo_topic[["sentence", "topic"]]
])

df_new_sent = pd.concat([
    df_labeled[["sentence", "sentiment"]],
    df_pseudo_sent[["sentence", "sentiment"]]
])

# === BƯỚC 7: Lưu kết quả ===
df_new_topic.to_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\expanded_topic.csv", index=False, encoding="utf-8-sig")
df_new_sent.to_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\expanded_sentiment.csv", index=False, encoding="utf-8-sig")

print("✅ Đã thêm", len(df_pseudo_topic), "pseudo-label cho topic")
print("✅ Đã thêm", len(df_pseudo_sent), "pseudo-label cho sentiment")
print("💾 File đã lưu: expanded_topic.csv & expanded_sentiment.csv")
