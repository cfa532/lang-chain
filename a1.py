import os
from apikey import API_KEY
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = API_KEY

# App framework
st.title('Whatever GPT creator')
prompt = st.text_input("Plug in your prompt here")

# LLMS
llm = OpenAI(temperature=0.9)

if prompt:
    res = llm(prompt=prompt)
    st.write(res)
