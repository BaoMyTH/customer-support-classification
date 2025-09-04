import pandas as pd

# Đọc dữ liệu đã gán nhãn (một phần)
df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\labeled_feedback.csv")

print("📌 Tổng số câu:", len(df))

# Tập labeled = những câu có cả topic và sentiment
df_labeled = df[(df["topic"].notna()) & (df["topic"] != "") &
                (df["sentiment"].notna()) & (df["sentiment"] != "")]

# Tập unlabeled = những câu chưa có nhãn
df_unlabeled = df.drop(df_labeled.index)

print("✅ Số câu có nhãn (labeled):", len(df_labeled))
print("✅ Số câu chưa nhãn (unlabeled):", len(df_unlabeled))

# Lưu ra file riêng để dễ dùng
df_labeled.to_csv("data/labeled_data.csv", index=False, encoding="utf-8-sig")
df_unlabeled.to_csv("data/unlabeled_data.csv", index=False, encoding="utf-8-sig")

print("💾 Đã lưu ra 2 file: labeled_data.csv và unlabeled_data.csv")
