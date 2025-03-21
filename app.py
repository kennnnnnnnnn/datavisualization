from dotenv import load_dotenv
import streamlit as st
import seaborn as sns
from langchain_anthropic import ChatAnthropic
from pandasai import SmartDataFrame

load_dotenv()

st.title("测试根据自然语言生成报表")

data = sns.load_dataset("penguins")

st.write(data.head(3))

model = ChatAnthropic(model="claude-3-haiku-20240307")
df = SmartDataFrame(data, config={"llm": model})

prompt = st.text_input("请输入自然语言描述")

if st.button("生成报表"):
    if prompt:
        with st.spinner("生成报表中..."):
            st.write(df.chat(prompt))

