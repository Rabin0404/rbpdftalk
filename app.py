import os
import chardet
import streamlit as st  # used to create our UI frontend
from apikey import apikey
from langchain.chat_models import ChatOpenAI  # used for GPT3.5/4 model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType

# os.environ["OPENAI_API_KEY"] = apikey

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title('Chat with Document')  # title in our web page
uploaded_file = st.file_uploader('Upload file:', type=['pdf', 'docx', 'txt'])
add_file = st.button('Add File', on_click=clear_history)

if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    file_name = os.path.join('./', uploaded_file.name)
    
    name, extension = os.path.splitext(file_name)
    
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        with open(file_name, 'wb') as f:
            f.write(bytes_data)
        loader = PyPDFLoader(file_name)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        with open(file_name, 'wb') as f:
            f.write(bytes_data)
        loader = Docx2txtLoader(file_name)
    elif extension == '.txt':
        detected_encoding = chardet.detect(bytes_data)['encoding']
        text_data = bytes_data.decode(detected_encoding)
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(text_data)
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_name, encoding='utf-8')
    else:
        st.write('Document format is not supported!')

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    # initialize OpenAI instance
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever = vector_store.as_retriever()
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    st.session_state.crc = crc
    # success message when file is chunked & embedded successfully
    st.success('File uploaded, chunked and embedded successfully')

    # Load tools and initialize agent
    tools = load_tools(['wikipedia', 'llm-math'], llm=llm)
    agent = initialize_agent(tools, llm, agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True)
    st.session_state.agent = agent

# get question from user input
question = st.text_input('Input your question')
if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
    if 'agent' in st.session_state:
        agent = st.session_state.agent
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # First, try to get the answer from the document
    response = crc.run({
        'question': question,
        'chat_history': st.session_state['history']
    })
    
    unsatisfactory_phrases = ["i don't know", "i'm not sure", "i don't have", "i don’t have that specific information"]
    if not response or any(phrase in response.strip().lower() for phrase in unsatisfactory_phrases):
        st.write("Using agent to find the answer...")
        response = agent.run(question)
    
    st.session_state['history'].append((question, response))
    st.write(response)
    for prompts in st.session_state['history']:
        st.write("Question: " + prompts[0])
        st.write("Answer: " + prompts[1])
