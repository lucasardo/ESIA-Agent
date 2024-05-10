
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

session_id = random.randint(0, 1000000)
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

    ### CREATE AGENT    
    prompt = AGENT_INTRO_PROMPT
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=10, reranker_th=1, sas_token='na')]
    COMPLETION_TOKENS = 4096
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature = 0)
   
    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )
    
    ### GENERATE RESPONSE
    response = with_message_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    
    history = update_history(session_id, question, response["output"], indexes)

    full_response = {
        "question": question,
        "output": response["output"],
        "history": history
    }

    response_text = full_response['output']
    response_intro = f"{response_text}"
    
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
    
    ### CREATE AGENT
    prompt = AGENT_ENV_PROMPT
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=10, reranker_th=1, sas_token='na')]
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature = 0)
    
    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
    )
    
    ### GENERATE RESPONSE
    response = with_message_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    history = update_history(session_id, question, response["output"], indexes)

    full_response = {
        "question": question,
        "output": response["output"],
        "history": history
    }

    response_text = full_response['output']
    response_env = f"{response_text}"
    
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

######################################
####### ENVIRONMENTAL ECONOMIST
######################################
                
    st.markdown('#')
    session_id = session_id + 2
    st.write("<h2 style='color: #F9423A;'>SOCIAL IMPACT", unsafe_allow_html=True)
    
    ### CREATE AGENT
    prompt = AGENT_SOCIAL_PROMPT
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=10, reranker_th=1, sas_token='na')]
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature = 0)

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
    )
    
    ### GENERATE RESPONSE
    response = with_message_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    history = update_history(session_id, question, response["output"], indexes)

    full_response = {
        "question": question,
        "output": response["output"],
        "history": history
    }

    response_text = full_response['output']
    response_social = f"{response_text}"
    
    st.markdown(response_social)
    
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
####### SUMMARIZER
######################################
                
    st.markdown('#')
    session_id = session_id+3
    st.write("<h2 style='color: #F9423A;'>CONCLUSION", unsafe_allow_html=True)
    
    ### CREATE AGENT
    prompt = AGENT_CONCLUSION_PROMPT
    tools = [GetDocSearchResults_Tool(indexes=indexes, k=10, reranker_th=1, sas_token='na')]
    llm = AzureChatOpenAI(deployment_name=openai_deployment_name, openai_api_version=openai_api_version,
                            openai_api_key=openai_api_key, azure_endpoint=azure_endpoint, temperature = 0)

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
    )
    
    ### GENERATE RESPONSE
    response = with_message_history.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

    history = update_history(session_id, question, response["output"], indexes)

    full_response = {
        "question": question,
        "output": response["output"],
        "history": history
    }

    response_text = full_response['output']
    response_conclusion = f"{response_text}"
    
    st.markdown(response_conclusion)

################################################################################################################
################################################################################################################

# DOWNLOAD WORD DOCUMENT
    
    save_as_word(response_intro, response_env, response_social, response_conclusion)
    with open("ESIA Draft.docx", "rb") as f:
        bytes_data = f.read()
    st.download_button(
        label="Click here to download",
        data=bytes_data,
        file_name="ESIA Draft.docx",
        mime="application/docx"
    )