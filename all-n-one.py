import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
import nltk
nltk.download('averaged_perceptron_tagger_eng')

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(api_key=api_key, model='gemma2-9b-it')

st.title('All in One: Summarizer')
st.info('Please Choose Type of Summarization')

radio_opt = ['Generic Text', 'PDF Document', 'YouTube Video', 'Website Page']
selected_opt = st.sidebar.radio('Select Type Of Summarization', radio_opt)



def preprocessing(pdf):
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
            
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        #add the loaded pdf to documents list
        documents.extend(docs)
        
    # Split and create embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    return splits

# Text Prompt
text_template = """Summarize the below text,
text:{text}"""

text_prompt = PromptTemplate(
    input_variables=['text'],
    template = text_template
)

# PDF Prompt
pdf_template = """Summarize the below document,
document:{text}"""

pdf_prompt = PromptTemplate(
    input_variables=['text'],
    template = pdf_template
)

# URL Prompt
url_template = """please provide summary to the following text in 300 words.
text:{text}"""

url_prompt = PromptTemplate(
    input_variables=['text'],
    template=url_template
)

final_prompt = """provide the final summary of the entire speech with these important points.
add a motivational title , start the percise summary with an introduction and provide the summary in numbers of points for the speech.
speech:{text}"""

final_prompt_template = PromptTemplate(
    input_variables=['text'],
    template = final_prompt
)


if radio_opt.index(selected_opt) == 1:
    uploaded_files = st.file_uploader('Choose a PDF file', type='pdf', accept_multiple_files=True)
    preprocess = preprocessing(uploaded_files)
    if uploaded_files:
        with st.spinner('Summarizing...'):

            chain = load_summarize_chain(llm, chain_type='stuff', prompt = pdf_prompt, verbose=True)
            st.success(chain.run(preprocess))
elif radio_opt.index(selected_opt)==0:
    text = st.text_area('Enter Text')
    if text:
        chain = LLMChain(llm = llm, prompt = text_prompt, verbose=True)
        st.success(chain.run(text))
elif radio_opt.index(selected_opt)==2:
    url = st.text_input('Enter YT URL' )
    if url:
        with st.spinner('Summarizing...'):
            if 'youtube.com' or 'youtube.be' in url:
                loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            docs =loader.load()
            chain = load_summarize_chain(llm, chain_type='stuff', prompt = url_prompt, verbose=True)
            st.success(chain.run(docs))
elif radio_opt.index(selected_opt)==3:
   
    url = st.text_input('Enter a Website URL' )
    if url:
        with st.spinner('Summarizing...'):
            loader = UnstructuredURLLoader(urls=[url], ssl_verify=True,headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
            docs = loader.load()
            chain = load_summarize_chain(llm, chain_type='stuff', prompt = url_prompt, verbose=True)
            st.success(chain.run(docs))
    

