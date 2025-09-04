import pandas as pd
import glob
import os

def merge_csv_in_folder(folder_path, output_file="merged.csv"):
    """
    Gộp tất cả file CSV trong một folder thành 1 file duy nhất.
    
    folder_path: đường dẫn tới folder chứa các file CSV
    output_file: tên file CSV đầu ra
    """
    # Lấy tất cả file CSV trong folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print("⚠️ Không tìm thấy file CSV nào trong folder.")
        return None
    
    print(f"🔎 Tìm thấy {len(csv_files)} file CSV trong folder.")
    
    # Đọc và gộp các file
    dfs = [pd.read_csv(f) for f in csv_files]
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Lưu file đầu ra
    output_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Đã gộp {len(csv_files)} file thành '{output_path}' với {len(merged_df)} dòng.")
    return merged_df

# Ví dụ sử dụng cho folder của bạn
folder = r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data"
merged = merge_csv_in_folder(folder, "feedback_merged.csv")
