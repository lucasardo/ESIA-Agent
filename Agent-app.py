from st_helper import *
from utils import *
from langchain_core.output_parsers import StrOutputParser

# Environment variables

azure_endpoint = st.secrets["azure_endpoint"]
openai_api_key = st.secrets['openai_api_key']

openai_deployment_name = st.secrets['openai_deployment_name']
openai_api_version = st.secrets['openai_api_version']
embedding_model = st.secrets['embedding_model']
embedding_deployment_name = st.secrets['embedding_deployment_name']

search_endpoint = st.secrets['search_endpoint']
search_api_key = st.secrets['search_api_key']
search_api_version = st.secrets['search_api_version']
search_service_name = st.secrets['search_service_name']

search_url = f"https://{search_service_name}.search.windows.net/"
search_credential = AzureKeyCredential(search_api_key)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"RAG Agent - Streamlit"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets['langsmith_key']

client = Client()

# Session variables

session_id = random.randint(0, 1000000)
index_name = "esias-base-index"

embed_model = AzureOpenAIEmbedding(
    model=embedding_model,
    deployment_name=embedding_deployment_name,
    api_key=openai_api_key,
    azure_endpoint=azure_endpoint,
    api_version=openai_api_version
)

llmChat = AzureOpenAI(
    engine=str(openai_deployment_name),
    azure_endpoint=str(azure_endpoint),
    api_key=str(openai_api_key),
    api_version=str(openai_api_version)
)

prompt_helper = PromptHelper(
    context_window=1536,
    num_output=256,
    chunk_overlap_ratio=0.1,
    chunk_size_limit=None,
)

service_context = ServiceContext.from_defaults(
            llm=llmChat,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
)

Settings.llm = llmChat
Settings.embed_model = embed_model

################################################################################################################
################################################################################################################

# Web page title
st.set_page_config(
    page_title="ESIA Agent 💬 WSP",
    page_icon=":robot:",
    layout="wide"
)

# Sidebar
st.sidebar.image("https://download.logo.wine/logo/WSP_Global/WSP_Global-Logo.wine.png", width=100)
if st.sidebar.button("New Thread"):
    st.experimental_rerun()
st.sidebar.markdown('#') 
st.sidebar.header("History")
st.sidebar.write("- I have to write an ESIA report for an offshore wind farm project, called 'ItaWind'. The wind farm will be located in Italy, off the coast of Molise.")

# Heading
if "typewriter_executed" not in st.session_state:
    st.session_state.typewriter_executed = False

header = "Hi, how can I help you today?"
subheader = "Dive into the world of environmental impact analysis with your AI guide, powered by WSP Digital Innovation!"

if not st.session_state.typewriter_executed:
    speed = 10
    typewriter_header(text=header, speed=speed)
    speed = 10
    typewriter_subheader(text=subheader, speed=speed)   
    st.session_state.typewriter_executed = True
else:
    st.markdown(f"<h1 style='color: #F9423A; text-align: center;'>{header}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color: #F9423A; text-align: center;'>{subheader}</h4>", unsafe_allow_html=True)
    
st.markdown('#') 

################################################################################################################
################################################################################################################

filter = ""

st.markdown('#') 
st.markdown('#') 

### USER INPUT

init_prompt = st.text_input("Ask anything")

### ANSWER GENERATION

if init_prompt:
    
    st.write("Request for Proposal:", init_prompt)
    question = str(init_prompt)
      
    indexes = [index_name]
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature=0)
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=2, reranker_th=0.02, sas_token='na')]
    
######################################
####### PROJECT MANAGER
######################################

    st.write(f"<h2 style='color: #F9423A;'>INTRODUCTION</h2>", unsafe_allow_html=True)

    response_intro = generate_intro(question, llm, tools, index_name, session_id)
    st.markdown(response_intro)
    
    ### RETRIEVE CITATIONS AND RRF SCORES
    search_results = simple_hybrid_search(question, index_name, filter, search_url, search_credential, azure_endpoint, openai_api_key, openai_api_version, embedding_deployment_name)
    sources_nodes = list_sources_nodes(search_results)
    
    with st.expander("See sources"):
        for node in sources_nodes[0:5]:
            file_name = os.path.basename(node["path"])
            st.write("Source file: ", file_name)
            raw_score = node["score"]
            score = "{:.2f}".format(raw_score)
            score = float(score)
            if score >= 0.02:
                st.write(":heavy_check_mark: High confidence. Search score: ", score)
            elif score >= 0.01:
                st.write(":warning: Medium confidence. Search score: ", score)
            else:
                st.write(":x: Low confidence. Search score: ", score)
        
######################################
####### ENVIRONMENTAL ENGINEER
######################################
                
    st.markdown('#')
    st.write(f"<h2 style='color: #F9423A;'>ENVIRONMENTAL IMPACT</h2>", unsafe_allow_html=True)
    
    response_env = generate_env_chapter(question, llm, tools, index_name, session_id)
    st.markdown(response_env)

    ### RETRIEVE CITATIONS AND RRF SCORES
    search_results = simple_hybrid_search(question, index_name, filter, search_url, search_credential, azure_endpoint, openai_api_key, openai_api_version, embedding_deployment_name)
    sources_nodes = list_sources_nodes(search_results)
 
    with st.expander("See sources"):
        for node in sources_nodes[0:5]:
            file_name = os.path.basename(node["path"])
            st.write("Source file: ", file_name)
            raw_score = node["score"]
            score = "{:.2f}".format(raw_score)
            score = float(score)
            if score >= 0.02:
                st.write(":heavy_check_mark: High confidence. Search score: ", score)
            elif score >= 0.01:
                st.write(":warning: Medium confidence. Search score: ", score)
            else:
                st.write(":x: Low confidence. Search score: ", score)
    
######################################
####### SOCIAL ECONOMIST
######################################
                
    st.markdown('#')
    st.write(f"<h2 style='color: #F9423A;'>SOCIAL IMPACT</h2>", unsafe_allow_html=True)
        
    response_social = generate_social_chapter(question, llm, tools, index_name, session_id)
    st.markdown(response_social)
    
    ### RETRIEVE CITATIONS AND RRF SCORES
    search_results = simple_hybrid_search(question, index_name, filter, search_url, search_credential, azure_endpoint, openai_api_key, openai_api_version, embedding_deployment_name)
    sources_nodes = list_sources_nodes(search_results)

    with st.expander("See sources"):
        for node in sources_nodes[0:5]:
            file_name = os.path.basename(node["path"])
            st.write("Source file: ", file_name)
            raw_score = node["score"]
            score = "{:.2f}".format(raw_score)
            score = float(score)
            if score >= 0.02:
                st.write(":heavy_check_mark: High confidence. Search score: ", score)
            elif score >= 0.01:
                st.write(":warning: Medium confidence. Search score: ", score)
            else:
                st.write(":x: Low confidence. Search score: ", score)
    
######################################
####### SUMMARIZER
######################################
                
    st.markdown('#')
    st.write(f"<h2 style='color: #F9423A;'>CONCLUSION</h2>", unsafe_allow_html=True)
    
    HISTORY_SUMMARY = str(chat_history[session_id])
    
    CONCLUSION_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", HISTORY_SUMMARY),
            ("human", "{question}"),
        ]
    )
    
    output_parser = StrOutputParser()
    
    chain = CONCLUSION_PROMPT | llm | output_parser
      
    response_conclusion = chain.invoke(
        {"question": "Write a conclusion chapter for an Environmental and Social Impact Assessment, using information provided in the message history."},
        config={"configurable": {"session_id": session_id}}
        )
    
    st.markdown(response_conclusion)

################################################################################################################
################################################################################################################
    
    ### GENERATE DOCX
    word_data = save_as_word(response_intro, response_env, response_social, response_conclusion)

    ### DOWNLOAD
    if word_data is None:
        st.error("Failed to generate the Word document.")
    else:
        # Convert to downloadable format
        b64_word_data = base64.b64encode(word_data).decode()

        # JavaScript to create a clickable download link
        download_link = f"""
            <a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64_word_data}" download="ESIA_Draft.docx" style="
                display: inline-block;
                padding: 8px 16px;
                font-size: 16px;
                color: #fff;
                background-color: #007bff;
                border: none;
                border-radius: 4px;
                text-align: center;
                text-decoration: none;
                cursor: pointer;
            ">
                Click here to download
            </a>
        """

        # Display the clickable download link in Streamlit
        st.markdown(download_link, unsafe_allow_html=True)

        ### CHAT HISTORY
    
    st.write(chat_history)
    st.write(store)