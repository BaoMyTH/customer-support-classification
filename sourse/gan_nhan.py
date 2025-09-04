import pandas as pd

# Đọc dữ liệu đã tách câu
df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\feedback_sentences.csv")

# Các nhãn topic và sentiment
topics = ["san_pham", "gia_ca", "dong_goi", "giao_hang", "mo_ta", "cskh", "trai_nghiem"]
sentiments = ["tich_cuc", "tieu_cuc", "trung_lap"]

# Thêm cột nhãn rỗng
df["topic"] = ""
df["sentiment"] = ""

print("=== BẮT ĐẦU GÁN NHÃN (NHẬP SỐ) ===")
print("Có tổng cộng:", len(df), "feedback sentences")

def show_menu(options):
    for i, opt in enumerate(options, start=1):
        print(f"{i}. {opt}")

# Vòng lặp gán nhãn
for i in range(len(df)):
    print("\n-----------------------------")
    print("Câu:", df.loc[i, "sentence"])
    
    # Gán topic
    print("Chọn topic:")
    show_menu(topics)
    t = input("Nhập số (1-7) hoặc Enter để bỏ qua: ").strip()
    if t.isdigit() and 1 <= int(t) <= len(topics):
        df.at[i, "topic"] = topics[int(t)-1]
    
    # Gán sentiment
    print("Chọn sentiment:")
    show_menu(sentiments)
    s = input("Nhập số (1-3) hoặc Enter để bỏ qua: ").strip()
    if s.isdigit() and 1 <= int(s) <= len(sentiments):
        df.at[i, "sentiment"] = sentiments[int(s)-1]
    
    # Lưu tạm mỗi 50 câu
    if i % 50 == 0 and i > 0:
        df.to_csv("labeled_feedback.csv", index=False, encoding="utf-8-sig")
        print("💾 Đã lưu tạm tại câu số", i)

# Lưu cuối cùng
df.to_csv("labeled_feedback.csv", index=False, encoding="utf-8-sig")
print("✅ Đã gán nhãn xong, lưu vào labeled_feedback.csv")
