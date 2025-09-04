import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def self_training_loop(
    labeled_file, unlabeled_file, max_iter=5,
    topic_threshold=0.6, sentiment_threshold=0.8,
    output_topic="expanded_topic.csv", output_sentiment="expanded_sentiment.csv"
):
    # Đọc dữ liệu
    df_labeled = pd.read_csv(labeled_file)
    df_unlabeled = pd.read_csv(unlabeled_file)

    print(f"🔄 Bắt đầu self-training (tối đa {max_iter} vòng)")
    print(f"   - Ngưỡng topic = {topic_threshold}, sentiment = {sentiment_threshold}")

    for it in range(max_iter):
        print(f"\n=== Vòng lặp {it+1} ===")

        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_labeled = vectorizer.fit_transform(df_labeled["sentence"])
        X_unlabeled = vectorizer.transform(df_unlabeled["sentence"])

        # Train mô hình topic
        clf_topic = LogisticRegression(max_iter=1000)
        clf_topic.fit(X_labeled, df_labeled["topic"])

        # Train mô hình sentiment
        clf_sentiment = LogisticRegression(max_iter=1000)
        clf_sentiment.fit(X_labeled, df_labeled["sentiment"])

        # --- Self-training cho Topic ---
        proba_topic = clf_topic.predict_proba(X_unlabeled)
        high_conf_idx = proba_topic.max(axis=1) > topic_threshold

        if high_conf_idx.any():
            pseudo_labels_topic = clf_topic.predict(X_unlabeled[high_conf_idx])
            df_pseudo_topic = df_unlabeled[high_conf_idx].copy()
            df_pseudo_topic["topic"] = pseudo_labels_topic
        else:
            df_pseudo_topic = pd.DataFrame(columns=["sentence", "topic"])

        # --- Self-training cho Sentiment ---
        proba_sent = clf_sentiment.predict_proba(X_unlabeled)
        high_conf_idx_s = proba_sent.max(axis=1) > sentiment_threshold

        if high_conf_idx_s.any():
            pseudo_labels_sent = clf_sentiment.predict(X_unlabeled[high_conf_idx_s])
            df_pseudo_sent = df_unlabeled[high_conf_idx_s].copy()
            df_pseudo_sent["sentiment"] = pseudo_labels_sent
        else:
            df_pseudo_sent = pd.DataFrame(columns=["sentence", "sentiment"])

        # Báo cáo
        print(f"✅ Vòng {it+1} thêm {len(df_pseudo_topic)} topic, {len(df_pseudo_sent)} sentiment")

        if len(df_pseudo_topic) == 0 and len(df_pseudo_sent) == 0:
            print("⏹ Không còn mẫu đủ tin cậy → dừng lại")
            break

        # Gộp vào tập labeled
        if not df_pseudo_topic.empty:
            df_pseudo_topic["sentiment"] = ""
            df_labeled = pd.concat([df_labeled, df_pseudo_topic[["sentence", "topic", "sentiment"]]])

        if not df_pseudo_sent.empty:
            df_pseudo_sent["topic"] = ""
            df_labeled = pd.concat([df_labeled, df_pseudo_sent[["sentence", "topic", "sentiment"]]])

        # Loại bỏ mẫu đã dùng
        df_unlabeled = df_unlabeled.drop(df_pseudo_topic.index, errors="ignore")
        df_unlabeled = df_unlabeled.drop(df_pseudo_sent.index, errors="ignore")

        # Lưu tạm
        df_labeled.to_csv(output_topic, index=False, encoding="utf-8-sig")
        df_labeled.to_csv(output_sentiment, index=False, encoding="utf-8-sig")
        print(f"💾 Đã lưu tạm sau vòng {it+1}")

    print("🎉 Self-training hoàn tất!")

# === Chạy thử ===
if __name__ == "__main__":
    self_training_loop(
        labeled_file=r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\labeled_data.csv",
        unlabeled_file=r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\unlabeled_data.csv",
        max_iter=5,
        topic_threshold=0.6,
        sentiment_threshold=0.8
    )
