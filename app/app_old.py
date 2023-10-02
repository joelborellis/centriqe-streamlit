import streamlit as st
import os
import openai


from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

st.title("Centriqe Product Assist")

with st.sidebar:
    "[Visit Centriqe](https://centriqe.com/)"
    material_options = st.multiselect(
            'What materials do you prefer?',
            ['Cotton', 'Rayon', 'Wool', 'Linen'])
    color_options = st.multiselect(
            'What colors do you prefer?',
            ['Blue', 'Green', 'Black', 'Olive'])
    fit_options = st.multiselect(
            'What fit do you prefer?',
            ['Slim', 'Regular'])


def search_api(query: str) -> str:
    model: str = "text-embedding-ada-002"

    vector_store_address: str = os.environ.get("AZURE_SEARCH_ENDPOINT")
    vector_store_key: str = os.environ.get("AZURE_SEARCH_ADMIN_KEY")

    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(deployment=model, chunk_size=1)
    index_name: str = "centriqe-vector-demo"

    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_key,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )

    # Perform a similarity search
    docs = vector_store.similarity_search(
        query=query,
        k=3,
        search_type="similarity",
    )
    #return (docs[0].page_content)
    #return "\n".join(docs.page_content)
    results = [doc.page_content for doc in docs]
    return "\n".join(results)

def generate_response(input_task, input_customer, input_material, input_color, input_fit):
    
    #openai.api_type = os.environ.get("OPENAI_API_TYPE")
    #openai.api_base = os.environ.get("OPENAI_API_BASE")
    #openai.api_version = os.environ.get("OPENAI_API_VERSION")
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    materials = [material for material in input_material]
    colors = [color for color in input_color]
    fits = [fit for fit in input_fit]
    print(" ".join(materials + colors + fits))

    # search based on the material color and fit and include the task
    r = search_api(" ".join(materials + colors + fits))
    #print(r)

    # Setup memory for contextual conversation
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", chat_memory=msgs, return_messages=True)

    prompt = PromptTemplate(
    input_variables=[
        "task",
        "product",
        "customer",
        "chat_history",
        "human_input", # Even if it's blank
    ],
    template=(
        """
        # MISSION
        Your goal is to generate meaningful and creative sales pitches that will be used by sales reps on customers in a store.

        # INTERACTION SCHEMA
        Users interacting with you will be sales reps who are in front of a customer and need a sales pitch based on the information provided to you below in #CONTEXT.
    
        Here is a specific task the sales rep is asking for:

        {task}

        #CONTEXT
        Use the following product information:
        
        {product}
        
        Combine this product information with the following user profile which might include preferences they have, information about their lifestyle, recent activities they did like travel or attend an event.

        {customer}

        # FORMAT
        Output your responses in a format that is easy for the user to read back to their customer.  Please add citation after each sentence when possible in a form "(Source: citation)".

        And here is extra human input:

        {human_input}

        Here is the chat history so far:

        {chat_history}
        """
    )
)

# Setup LLM and QA chain
    llm = ChatOpenAI(
        openai_api_key=openai.api_key, model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True
    )

    #prompt_template = "{question}.  Find me information about the following accounts {account} in the following industry {industry}?"

    llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory, # Contains the input_key
            verbose=True
        )
    #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    st.info(llm_chain.run({'human_input': "", 'task': input_task, 'product': r, 'customer': input_customer}))


with st.form("my_form"):
    #product = st.selectbox("Product: ", ('Greenfibre Mens Dark Grey Terry Rayon Slim Fit Solid Formal Trouser', 'Modi Jacket Mens Orange Terry Wool Textured Regular Fit Jacket', 'Greenfibre Mens Vineyard Green 100% Cotton Slim Fit Checked Casual Shirt'))
    customer = st.text_area("Customer details:", "Loves the outdoors, likes to run and hike")
    task = st.text_area("Perform task:", "Write me a suggestion for a formal shirt combined with formal trousers")
    
    submitted = st.form_submit_button("Submit")

    st.divider()
    
    #if not openai_api_key:
    #    st.info("Please add your OpenAI API key to continue.")
    if submitted:
        with st.spinner('Calling API...'):
            generate_response(task, customer, material_options, color_options, fit_options)
            st.success('Done!')
