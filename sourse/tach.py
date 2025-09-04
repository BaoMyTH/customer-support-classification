import pandas as pd
import re
from underthesea import sent_tokenize

df = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data\feedback_merged.csv")

# ==== BƯỚC 2: HÀM TÁCH CÂU ====
def custom_sentence_split(text):
    """pip
    Tách câu/ý nhỏ từ feedback
    1. Dùng underthesea để tách câu chuẩn
    2. Sau đó tách thêm theo dấu phẩy, từ nối ('nhưng', 'tuy nhiên')
    """
    if pd.isna(text):
        return []
    
    # Tách câu ban đầu bằng underthesea
    sentences = sent_tokenize(text)
    results = []
    
    for sent in sentences:
        # Tách tiếp theo từ nối và dấu phẩy
        subs = re.split(r',| nhưng | tuy nhiên | mà | và ', sent, flags=re.IGNORECASE)
        for s in subs:
            s = s.strip()
            if s:  # loại bỏ chuỗi rỗng
                results.append(s)
    
    return results

# ==== BƯỚC 3: TÁCH CÂU TOÀN BỘ DATA ====
rows = []
for idx, text in enumerate(df['content']):
    sentences = custom_sentence_split(text)
    for sent in sentences:
        rows.append({"id": idx, "sentence": sent})

df_sentences = pd.DataFrame(rows)

# ==== BƯỚC 4: LƯU KẾT QUẢ ====
df_sentences.to_csv("feedback_sentences.csv", index=False, encoding="utf-8-sig")

print("✅ Hoàn thành! Đã lưu file feedback_sentences.csv")
