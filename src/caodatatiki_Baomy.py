import requests
import pandas as pd
import time

product_id = "185860447"  
all_reviews = []

for page in range(1, 11):  # Crawl 10 trang đầu
    url = f"https://tiki.vn/api/v2/reviews?product_id={product_id}&limit=20&page={page}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for review in data['data']:
            all_reviews.append({
                "content": review.get("content"),
                "rating": review.get("rating"),
                "created_at": review.get("created_by", {}).get("name"),
                "thank_count": review.get("thank_count")
            })
    else:
        print(f"Lỗi trang {page}")
    time.sleep(1)

    df = pd.DataFrame(all_reviews)
df.to_csv("data_raw/Baomy_tiki_5.csv", index=False, encoding="utf-8-sig")
print("Đã lưu xong!")