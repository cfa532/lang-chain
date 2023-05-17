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
title_template = PromptTemplate(
    input_variables=['topic'],
    template="What is the fastest car in {topic}"
)
script_template = PromptTemplate(
    input_variables=['title', 'wiki_research'],
    template="Write me a Youtube script about {title} and leverage this Wiki research:{wiki_research}"
)
# Memory
title_mem = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_mem = ConversationBufferMemory(input_key="title",  memory_key="chat_history")
wiki = WikipediaAPIWrapper() 

# LLMS
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_mem)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_mem)
seq_chain  = SequentialChain(chains=[title_chain, script_chain], input_variables=["topic", "wiki_research"], 
                             output_variables=["title", "script"], verbose=True)
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,  wiki_research=wiki_research)
    # response = seq_chain({"topic": prompt})
    st.write(title)
    st.write(script)

    with st.expander("Message history"):
        st.info(title_mem.buffer)
    with st.expander("Script history"):
        st.info(script_mem.buffer)
    with st.expander("Wiki research"):
        st.info(wiki_research)