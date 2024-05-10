
################################################################################################################

from st_helper import *

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

# Session variables

session_id = 3442
index_name = "esias-base-index"
max_tokens = 4096
dimensionality = 1536

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
    
    indexes = [index_name]

#######################
####### PROJECT MANAGER
#######################

    st.write("<h2 style='color: #F9423A;'>INTRODUCTION", unsafe_allow_html=True)

    question = str(init_prompt)  
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature=0)
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=3, reranker_th=0, sas_token='na')]

    response_intro = generate_intro(question, llm, tools, index_name, session_id)
    
    st.markdown(response_intro)
    
    ### RETRIEVE CITANTIONS AND RRF SCORES
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
    session_id = session_id + 1
    st.write("<h2 style='color: #F9423A;'>ENVIRONMENTAL IMPACT", unsafe_allow_html=True)
    
    @st.cache(suppress_st_warning=True)
    def cached_env_chapter(question, index_name, session_id):
        return generate_env_chapter(question, index_name, session_id)      
    
    response_env = cached_env_chapter(question, index_name, session_id)
    
    st.markdown(response_env)

    ### RETRIEVE CITANTIONS AND RRF SCORES
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

