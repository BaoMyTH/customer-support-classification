import pandas as pd

df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_clean\cleaned_data.csv")

df["sentiment"] = None   # tạo cột label rỗng

for i, row in df.iterrows():
    # In ra nội dung feedback để bạn đọc và gán nhãn
    print(f"\n--- Feedback {i} ---")
    print(f"Content: {row['content']}") # Thay 'content' bằng tên cột chứa văn bản feedback của bạn
    label = input("Nhập nhãn cho feedback này (ví dụ: positive, negative, neutral): ")
    df.at[i, "sentiment"] = label

# Sau khi gắn nhãn xong, lưu lại file
print("\nĐã hoàn thành việc gán nhãn. Đang lưu file...")
df.to_csv("data_clean/sentiment.csv", index=False)
print("File 'sentiment.csv' đã được lưu thành công!")

