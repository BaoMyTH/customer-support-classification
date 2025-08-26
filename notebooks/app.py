# app.py

import streamlit as st
import joblib
import re
import pandas as pd

# --- 1. Hàm tiền xử lý văn bản ---
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

# --- 2. Cấu hình Streamlit ---
st.set_page_config(page_title="Hệ thống Phân loại Phản hồi & Phản hồi Tự động", layout="centered")
st.title("🤖 Hệ thống Phân loại Phản hồi & Phản hồi Tự động")
st.markdown("""
Chào mừng bạn đến với hệ thống phân tích phản hồi khách hàng.
Nhập phản hồi của bạn vào ô bên dưới để nhận được phân loại cảm xúc, chủ đề và một phản hồi tự động!
""")

# --- 3. Hàm tải mô hình ---
@st.cache_resource
def load_models():
    st.info("Đang tải các mô hình đã huấn luyện...")
    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        sentiment_model = joblib.load('sentiment_model.pkl')
        topic_model = joblib.load('topic_model.pkl')
        st.success("Tải mô hình hoàn tất!")
        return vectorizer, sentiment_model, topic_model
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy file mô hình. Vui lòng chạy 'train_and_save_models.ipynb' trước!")
        st.stop()

vectorizer, sentiment_model, topic_model = load_models()

# --- 4. Phản hồi tự động ---
auto_responses = {
    'sentiment': {
        'p': "Cảm ơn bạn đã phản hồi tích cực! Chúng tôi rất vui vì bạn hài lòng.",
        'n': "Chúng tôi rất tiếc về trải nghiệm không tốt của bạn. Vui lòng cho chúng tôi biết thêm chi tiết để chúng tôi có thể hỗ trợ tốt hơn."
    },
    'label': {
        'quality': "Cảm ơn bạn đã đánh giá sản phẩm. Chúng tôi luôn cố gắng cải thiện chất lượng sản phẩm.",
        'other': "Chúng tôi đã nhận được yêu cầu về dịch vụ khách hàng của bạn. Một nhân viên sẽ liên hệ lại sớm nhất có thể để hỗ trợ.",
        'deliver': "Chúng tôi sẽ kiểm tra tình trạng đơn hàng giao hàng của bạn ngay lập tức và cập nhật thông tin.",
        'pay': "Về vấn đề thanh toán/hóa đơn, vui lòng cung cấp mã đơn hàng hoặc thông tin liên quan để chúng tôi kiểm tra chi tiết.",
        'hdsd': "Chúng tôi hiểu bạn đang gặp sự cố về việc sử dụng sản phẩm. Vui lòng mô tả chi tiết lỗi để được hỗ trợ nhanh nhất từ đội ngũ kỹ thuật.",
        'baohanhdt': "Cảm ơn câu hỏi của bạn. Chúng tôi sẽ cung cấp thông tin chi tiết bạn cần hoặc chuyển đến bộ phận chuyên trách.",
        'promo': "Chúng tôi đã nhận được yêu cầu đổi trả sản phẩm của bạn. Vui lòng kiểm tra email để biết hướng dẫn chi tiết về quy trình."
    }
}

# --- 5. Hàm phân loại & tạo phản hồi ---
def generate_auto_response(text_input, vectorizer_obj, sentiment_model_obj, topic_model_obj, responses_dict):
    text_input_clean = clean_text(text_input)
    if not text_input_clean.strip():
        return "", "", "Vui lòng nhập phản hồi để phân tích.", "Vui lòng nhập phản hồi để phân tích."

    text_tfidf = vectorizer_obj.transform([text_input_clean])
    predicted_sentiment = sentiment_model.predict(text_tfidf)[0]  
    predicted_topic = topic_model.predict(text_tfidf)[0].lower()     

    st.write("DEBUG:", predicted_sentiment, predicted_topic)

    sentiment_response = responses_dict['sentiment'].get(predicted_sentiment, "Cảm ơn phản hồi của bạn.")
    topic_response = auto_responses['label'].get(predicted_topic, "Chúng tôi đã nhận được yêu cầu của bạn.")

    return predicted_sentiment, predicted_topic, sentiment_response, topic_response

# --- 6. Giao diện người dùng ---
user_feedback = st.text_area(
    "Nhập phản hồi của khách hàng vào đây:", 
    height=150,
    placeholder="Ví dụ: Sản phẩm rất tốt, tôi rất hài lòng với chất lượng!"
)

if st.button("Phân tích và Tạo Phản hồi"):
    pred_sentiment, pred_topic, sentiment_res, topic_res = generate_auto_response(
        user_feedback, vectorizer, sentiment_model, topic_model, auto_responses
    )

    st.subheader("Kết quả Phân tích:")
    st.info(f"**Cảm xúc Dự đoán:** `{pred_sentiment.upper()}`")
    st.info(f"**Chủ đề Dự đoán:** `{pred_topic.replace('_', ' ').title()}`")

    st.subheader("Phản hồi Tự động Gợi ý:")
    st.success(f"**Phản hồi dựa trên Chủ đề:** {topic_res}")
    st.success(f"**Phản hồi dựa trên Cảm xúc:** {sentiment_res}")
    st.markdown("---")
    st.write(f"**Phản hồi Tổng hợp:** {topic_res} {sentiment_res}")

