from utils import *
from prompts import *

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

index_name = "esias-base-index"

########################################### DESIGN ###################################################

def typewriter_header(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(f"<h1 style='color: #F9423A; text-align: center;'>{curr_full_text}</h1>", unsafe_allow_html=True)
        time.sleep(1 / speed)

def typewriter_subheader(text: str, speed: int):
    tokens = text.split()
    container = st.empty()
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(f"<h4 style='color: #F9423A; text-align: center;'>{curr_full_text}</h4>", unsafe_allow_html=True)
        time.sleep(1 / speed)

########################################### HYBRID SEARCH ###################################################

def get_embeddings(text, azure_endpoint, api_key, api_version, embedding_deployment_name):

    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    ) 
    embedding = client.embeddings.create(input=[text], model=embedding_deployment_name)
    return embedding.data[0].embedding

def simple_hybrid_search(query, index_name, filter, search_url, search_credential, azure_endpoint, openai_api_key, openai_api_version, embedding_deployment_name):

    search_client = SearchClient(
            endpoint=search_url,
            index_name=index_name,
            credential=search_credential,
        )
    
    vector_query = VectorizedQuery(vector=get_embeddings(query, azure_endpoint, openai_api_key, openai_api_version, embedding_deployment_name), k_nearest_neighbors=3, fields="embedding")

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        select=["chunk", "doc_path", "energy_sector"],
        filter=filter,
        top=5
    )

    return results

########################################### DISPLAY CHUNK SOURCES ###################################################

def list_sources_nodes(search_results):
    sources_nodes = []
    
    for result in search_results:
        score = result["@search.score"]
        path = result["doc_path"]
        node = {"path": path, "score": score}
        sources_nodes.append(node)
        
    return sources_nodes

########################################### SEARCH TOOLS ###################################################

class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

def get_search_results(query: str, indexes: list,
                       k: int = 10,
                       reranker_threshold: int = 1,
                       sas_token: str = "") -> List[dict]:
    """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""
    
    # Define the request headers
    headers = {
        "Content-Type": "application/json",
        "api-key": search_api_key  # Replace with your actual API key
    }

    params = {'api-version': search_api_version}

    k = 1
    
    agg_search_results = dict()

    # Define the request payload
    search_payload = {
        "search": query,
        "select": "id, doc_path, energy_sector, chunk",
        "vectorQueries": [{"kind": "text", "k": k, "fields": "embedding", "text": query}],
        "count": "true",
        "top": k
    }
    
    response = requests.post(search_endpoint + "indexes/" + index_name + "/docs/search",
                         data=json.dumps(search_payload), headers=headers, params=params)

    search_results = response.json()
    agg_search_results[index_name] = search_results

    reranker_threshold = 0

    content = dict()
    ordered_content = OrderedDict()

    for index, search_results in agg_search_results.items():
        for result in search_results['value']:
            # Show results that are at least N% of the max possible score=4
            if result['@search.score'] > reranker_threshold:
                content[result['id']] = {
                    "title": result['energy_sector'],
                    "name": result['energy_sector'],
                    "chunk": result['chunk'],
                    "location": result['doc_path'],
                    # "caption": result['@search.captions'][0]['text'],
                    "score": result['@search.score'],
                    "index": index
                }

    topk = k

    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding topK results
            break

    return ordered_content   

class CustomAzureSearchRetriever(BaseRetriever):

    indexes: List
    topK: int
    reranker_threshold: int
    sas_token: str = ""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        ordered_results = get_search_results(
            query, self.indexes, k=self.topK, reranker_threshold=self.reranker_threshold, sas_token=self.sas_token)

        top_docs = []
        for key, value in ordered_results.items():
            location = value["location"] if value["location"] is not None else ""
            try:
                top_docs.append(Document(page_content=value["chunk"], metadata={
                    "source": location, "score": value["score"]}))
            except:
                print("An exception occurred")
 
        # print(top_docs) 

        return top_docs  

class GetDocSearchResults_Tool(BaseTool):
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch"
    args_schema: Type[BaseModel] = SearchInput

    indexes: List[str] = []
    k: int = 10
    reranker_th: int = 1
    sas_token: str = ""

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:

        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, reranker_threshold=self.reranker_th,
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        results = retriever.get_relevant_documents(query=query)
        
        return results

########################################### AGENTS ###################################################

AGENT_INTRO_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CUSTOM_CHATBOT_PREFIX + PROMPT_TEMPLATE_INTRO),
        MessagesPlaceholder(variable_name='history', optional=True),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

AGENT_ENV_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CUSTOM_CHATBOT_PREFIX + PROMPT_TEMPLATE_ENV),
        MessagesPlaceholder(variable_name='history', optional=True),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

AGENT_SOCIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CUSTOM_CHATBOT_PREFIX + PROMPT_TEMPLATE_SOCIAL),
        MessagesPlaceholder(variable_name='history', optional=True),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

AGENT_CONCLUSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CUSTOM_CHATBOT_PREFIX + PROMPT_TEMPLATE_CONCLUSION),
        MessagesPlaceholder(variable_name='history', optional=True),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name='agent_scratchpad')
    ]
)

########################################### CHAT HISTORY ###################################################

store = {}
chat_history = {}
    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def update_history(session_id, human_msg, ai_msg, indexes):
    if session_id not in chat_history:
        chat_history[session_id] = []
        
    chat_history[session_id].append({
        "question": human_msg, 
        "output": ai_msg, 
        "indexes": indexes
    })
    return chat_history[session_id]

########################################### ANSWER GENERATION ###################################################

def generate_intro(question, llm, tools, indexes, session_id):

    prompt = AGENT_INTRO_PROMPT

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

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

    return response_intro


def generate_env_chapter(question, llm, tools, indexes, session_id):

    prompt = AGENT_ENV_PROMPT

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

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

    return response_env


def generate_social_chapter(question, llm, tools, indexes, session_id):

    prompt = AGENT_SOCIAL_PROMPT

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

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

    return response_social


def generate_conclusion(question, llm, tools, indexes, session_id):

    prompt = AGENT_CONCLUSION_PROMPT

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=False)

    with_message_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

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

    return response_conclusion

########################################### WORD DOCX GENERATION ###################################################

def save_as_word(response_1, response_2, response_3, response_4):
    doc = Docx()

    def add_content_with_headings(content):
        lines = content.split('\n')
        for line in lines:
            if line.startswith('####'):
                heading = doc.add_heading(line.lstrip('# '), level=2)
                for run in heading.runs:
                    run.font.color.rgb = RGBColor(255, 0, 0)
            elif line.startswith('###'):
                heading = doc.add_heading(line.lstrip('# '), level=1)
                for run in heading.runs:
                    run.font.color.rgb = RGBColor(255, 0, 0)
            else:
                doc.add_paragraph(line)
        doc.add_paragraph()

    # Add content to the document
    add_content_with_headings(response_1)
    add_content_with_headings(response_2)
    add_content_with_headings(response_3)
    add_content_with_headings(response_4)

    # Save the document to a bytes buffer
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)  # Ensure the buffer's pointer is at the beginning
    return buffer.getvalue()