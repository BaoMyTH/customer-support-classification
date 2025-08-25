import pandas as pd

df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\Baomy_tiki_5.csv")

df["label"] = None   # tạo cột label rỗng

for i, row in df.iterrows():
    print(f"\nFeedback {i}: {row['content']}")
    label = input("Nhập nhãn cho feedback này: ")
    df.at[i, "label"] = label

# Sau khi gắn nhãn xong, lưu lại file
df.to_csv("data_raw/labeled_Baomy_tiki_5.csv", index=False)
