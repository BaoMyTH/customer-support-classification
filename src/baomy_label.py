import pandas as pd

df_lb1 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\Baomy_tiki_1.csv")
df_lb2 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\Baomy_tiki_2.csv")
df_lb3 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\noisy_comments.csv")
df_lb4 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\Baomy_tiki_4.csv")
df_lb5 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\labeled_Baomy_tiki_5.csv")
df_lb6 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\noisy_feedback_payment.csv")
df_lb6 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\noisy_feedback_promo.csv")
df_lb7 = pd.read_csv(r"C:\Users\admin\Desktop\Git_p\customer-support-classification\data_raw\noisy_feedback.csv")


dfmylabel = pd.concat([df_lb1, df_lb2, df_lb3,df_lb4,df_lb5,df_lb6,df_lb7], ignore_index=True)


dfmylabel["label"] = None   # tạo cột label rỗng

for i, row in dfmylabel.iterrows():
    print(f"\nFeedback {i}: {row['content']}")
    label = input("Nhập nhãn cho feedback này: ")
    dfmylabel.at[i, "label"] = label

# Sau khi gắn nhãn xong, lưu lại file
dfmylabel.to_csv("data_clean/topic_Baomy_tiki.csv", index=False)