import pandas as pd

# Đọc file csv của bạn
df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\noisy_comments.csv")

# Gán nhãn "neg" cho toàn bộ data
df["label"] = "neg"

# Lưu lại ra file mới
df.to_csv("data_raw/Baomy_labeled_3.csv", index=False, encoding="utf-8")

print("Đã gán nhãn 'neg' cho toàn bộ dữ liệu và lưu vào noisy_comments_labeled.csv")
