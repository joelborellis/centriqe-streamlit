import openai
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import OpenAIEmbeddings
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField
)
import re

load_dotenv()

vector_store_address: str = os.environ.get("AZURE_SEARCH_ENDPOINT")
vector_store_key: str = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
model_embed: str = os.environ.get("OPENAI_MODEL_EMBED")
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model_embed, chunk_size=1)

index_name: str = os.environ.get("AZURE_SEARCH_INDEX")

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_key,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    semantic_configuration_name='centriqe-config',
        semantic_settings=SemanticSettings(
            default_configuration='centriqe-config',
            configurations=[
                SemanticConfiguration(
                    name='centriqe-config',
                    prioritized_fields=PrioritizedFields(
                        title_field=SemanticField(field_name='content'),
                        prioritized_content_fields=[SemanticField(field_name='content')],
                        prioritized_keywords_fields=[SemanticField(field_name='metadata')]
                    ))
            ])
    )

with st.sidebar:
    #st.button("New Chat", type="primary")
    st.write("💬 Start a new Chat")
    if st.button('New Chat'):
        # Delete all the items in Session state
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    st.subheader('README', divider='blue')
    st.write('The chat function is being driven by a system prompt and :blue[GPT-4] - it is using information from a product catalog to generate questions. :red[Improvement:] - add user infromation and more rich product data')
    st.write('After chatting and clicking :red[Done] there is another system prompt and a call to :blue[GPT-4] to generate the notes.')
    st.write('After notes are generated, they are passed to another system prompt and a call to :blue[GPT-4] to generate the suggestions.  :red[Improvement:] - need more rich product catalog')
    st.write('The system prompt used to generate the suggestions uses the notes and context data from a product catalog.  :red[Improvement:] - need more rich product catalog')

st.title("Centriqe - AskBuddy")

with st.expander("⚙️ Settings"):
    with st.container():
        st.write("LLM Parameters (for chat only)")
        tab1, tab2, tab3 = st.tabs(["Temperature", "Max Tokens", "Model"])
        tab1.subheader("Set Temperature")
        temp = tab1.slider('Temperature', 0, 1, 0)
        tab2.subheader("Set Max Tokens")
        max_tokens = tab2.slider('Max Tokens', 0, 4000, 2000)
        tab3.subheader("Choose a model")
        model = tab3.selectbox(
            'Choose Model',
            ('gpt-4-32k-0613', 'gpt-35-turbo-16k-0613'))
st.divider()

###     file operations

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
        return infile.read()
    
def chatbot(conversation, engine=model, temperature=0, max_tokens=2000):
    max_retry = 7
    retry = 0    
    print(model)
    while True:
        try:
            response = openai.ChatCompletion.create(engine=engine, messages=conversation, temperature=temperature, max_tokens=max_tokens)
            text = response['choices'][0]['message']['content']
            
            return text, response['usage']['total_tokens']
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            exit(5)

def find_url(string):  
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  
    found_urls = re.findall(url_pattern, string)  
    return found_urls  

def calculate_total_cost(cost_per_token, num_tokens):  
    return cost_per_token * num_tokens 

def search_api(query: str) -> str:
    results = []
    # Perform a similarity search
    docs = vector_store.similarity_search(
        query=query,
        k=10,
        search_type="hybrid",
        )

    for doc in docs:
        results.append(f"[PRODUCT NAME:  {doc.metadata['name']}]" + f"[PRODUCT BRAND:  {doc.metadata['brand']}]" + "\n" + f"[URL:  {doc.metadata['image']}]" + "\n" + f"[PRICE:  {doc.metadata['price']}]" + "\n" + (doc.page_content))
    #print("\n".join(results))
    return ("\n".join(results))
    
# use this for capturing an easy to read transcript of the chat conversation
all_messages = list()

if "messages_chatbot" not in st.session_state:
    st.session_state["messages_chatbot"] = [{"role": "assistant", "content": "Please tell me what you are looking for?"}]

if "completion_tokens" not in st.session_state:
    st.session_state["completion_tokens"] = 0

# display the flow of chat bubbles between user and assistant
for msg in st.session_state.messages_chatbot:
    if msg["role"] == "user":
        st.chat_message(msg["role"]).write(msg["content"])
        all_messages.append('SELLER: %s' % msg["content"])
    elif msg["role"] == "assistant":
        st.chat_message(msg["role"]).write(msg["content"])
        all_messages.append('INTAKE: %s' % msg["content"])

if prompt := st.chat_input():

    openai.api_type = os.environ.get("OPENAI_API_TYPE")
    openai.api_base = os.environ.get("OPENAI_API_BASE")
    openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # set the system message and user message into the session state
    st.session_state.messages_chatbot.append({"role": "system", "content": open_file('../system_01_intake.md').replace('<<CONTEXT>>', search_api(prompt))})
    st.session_state.messages_chatbot.append({"role": "user", "content": prompt})

    # now fill the user chat bubble and append to all_messages - this is used to make an easy to read conversation
    st.chat_message("user").write(prompt)

    # now call the openai chat completion and get the response into a string
    #msg, tokens = chatbot(st.session_state.messages_chatbot)
    print(model)
    response = openai.ChatCompletion.create(engine=model, temperature=temp, max_tokens=max_tokens, messages=st.session_state.messages_chatbot)
    tokens = response['usage']['total_tokens']
    #print(tokens)
    st.session_state.completion_tokens = tokens
    msg = response.choices[0].message

    # add the response from the llm to the session state
    st.session_state.messages_chatbot.append(msg)

    # now fill the assistant chat bubble and append to all_messages - this is used to make an easy to read conversation
    st.chat_message("assistant").write(msg.content)

st.divider()

with st.form("send_intake"):
        st.write("Click to start generating notes and suggestions")
        submitted = st.form_submit_button("Done")
        if submitted:
            conversation = list()
            conversation.append({'role': 'system', 'content': open_file('../system_02_prepare_notes.md')})
            text_block = '\n\n'.join(all_messages)
            chat_log = '<<BEGIN SELLER INTAKE CHAT>>\n\n%s\n\n<<END SELLER INTAKE CHAT>>' % text_block
            #print(st.session_state.completion_tokens)
            with st.expander("💬 Chat Log"):
                st.write(chat_log)
                #st.write(calculate_total_cost(.12, st.session_state.completion_tokens))
            #save_file('logs/log_%s_chat.txt' % time(), chat_log)
            conversation.append({'role': 'user', 'content': chat_log})
            with st.spinner('Creating notes...'):
                notes, tokens = chatbot(conversation, model)
            #print('\n\nNotes version of conversation:\n\n%s' % notes)
            #save_file('logs/log_%s_notes.txt' % time(), notes)
            with st.expander("🗒️ Notes"):
                st.write(notes)

            #print('\n\nGenerating Hypothesis Report')
            conversation = list()
            conversation.append({'role': 'system', 'content': open_file('../system_03_suggestion.md').replace('<<CONTEXT>>', search_api(notes))})
            conversation.append({'role': 'user', 'content': notes})
            with st.spinner('Creating suggestions...'):
                report, tokens = chatbot(conversation, model)
            #url = find_url(report)  
            #print(url)  
            with st.expander("🗣️ Suggestions"):
                st.write(report)
