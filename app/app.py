import streamlit as st
import os
import openai

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()

st.title("🦜🔗 Langchain Quickstart App")

with st.sidebar:
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"


def generate_response(input_industry, input_account, input_text):
    #openai.api_type = os.environ.get("OPENAI_API_TYPE")
    #openai.api_base = os.environ.get("OPENAI_API_BASE")
    #openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = OpenAI(
        openai_api_key=openai.api_key, model="gpt-35-turbo-16k-0613", temperature=0, streaming=True
    )

    prompt_template = "{question}.  Find me information about the following accounts {account} in the following industry {industry}?"

    llm_chain = LLMChain(
        memory=memory,
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    st.info(llm_chain.run({'question': input_text, 'account': input_account, 'industry': input_industry}))


with st.form("my_form"):
    industry = st.selectbox("What is the industry: ", ('Insurance', 'Oil and Gas', 'Technology'))
    account = st.text_input("Account:", type="default")
    text = st.text_area("Enter text:", "Help me with a sales situation?")
    submitted = st.form_submit_button("Submit")
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    if submitted:
        generate_response(industry, account, text)
