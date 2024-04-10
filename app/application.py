import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
pipe_lr=joblib.load(open("models/emotion_classifier_pipe_lr_08_june_2023.pkl","rb"))
def predict_emotions(docx):
    results=pipe_lr.predict([docx])
    return results[0]
def get_predic_prob(docx):
    results=pipe_lr.predict_proba([docx])
    return results
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Main Application
def main():
    st.title("INSAANI MAANSIKTA")
    menu=["home","monitor","about"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=="home":
        st.subheader("home-emotion in text")
        with st.form(key='tarun'):
            raw_text=st.text_area("Type here")
            submit_text=st.form_submit_button(label='Submit')
        if submit_text:
            col1, col2 = st.columns(2)
            prediction=predict_emotions(raw_text)
            probability=get_predic_prob(raw_text)


            with col1:
                st.success("originl text")
                st.write(raw_text)
                st.success("prediction")
                emoji_icon=emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("confidence:{}".format(np.max(probability*100)))
            with col2:
                st.success("prediction probbility")
                #st.write(probability)
                proba_df=pd.DataFrame(probability,columns=pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions","probability"]
                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                st.altair_chart(fig,use_container_width=True)


    elif choice=="monitor":
        st.subheader("monitor app")
    else:
        st.subheader("about")
if __name__=='__main__':
    main()
