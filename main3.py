import streamlit as st
from streamlit_chat import message
from langchain import LLMMathChain, OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import DuckDuckGoSearchRun
import requests
import json
import base64
from PIL import Image

def asticaAPI(endpoint, payload, timeout):
    response = requests.post(endpoint, data=json.dumps(payload), timeout=timeout, headers={ 'Content-Type': 'application/json', })
    if response.status_code == 200:
        return response.json()
    else:
        return {'status': 'error', 'error': 'Failed to connect to the API.'}

def analyze_image(asticaAPI_key, image_url):
    asticaAPI_timeout = 200 # seconds  Using "gpt" or "gpt_detailed" will increase response time.
    asticaAPI_endpoint = 'https://vision.astica.ai/describe'
    asticaAPI_modelVersion = '2.1_full' # '1.0_full', '2.0_full', or '2.1_full'
    asticaAPI_input = image_url
    asticaAPI_visionParams = 'gpt,description,objects,faces' # comma separated options; leave blank for all; note "gpt" and "gpt_detailed" are slow.

    # Define payload dictionary
    asticaAPI_payload = {
        'tkn': asticaAPI_key,
        'modelVersion': asticaAPI_modelVersion,
        'visionParams': asticaAPI_visionParams,
        'input': asticaAPI_input,
    }

    # Call API function and store result
    asticaAPI_result = asticaAPI(asticaAPI_endpoint, asticaAPI_payload, asticaAPI_timeout)

    return asticaAPI_result

def init():
    st.set_page_config(
        page_title="SafiraGPT 🤖",
        page_icon="🤖",
    )

    st.header('SafiraGPT 🤖')

def main():
    init()

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    asticaAPI_key = st.sidebar.text_input("Astica API Key", type="password")

    if not openai_api_key or not asticaAPI_key:
        st.info("Por favor, insira suas chaves API da OpenAI e Astica para continuar.")
        st.stop()

    # Crie uma instância do chatbot com o modelo desejado
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    # Crie uma instância do DuckDuckGoSearchRun
    ddg_search = DuckDuckGoSearchRun()

    # Crie as ferramentas que você quer adicionar ao seu chatbot
    tools = [
        Tool(
            name = "Search",
            func=ddg_search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions, and search time, open link and describe, etc."
        ),
        Tool(
            name = "Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool(
            name = "ImageAnalysis",
            func=lambda image_url: analyze_image(asticaAPI_key, image_url),
            description="useful for when you need to analyze an image, just translate the result"
        )
    ]

    system_message = SystemMessage(content="VOCÊ É UM SUPER ASSISTENTE VIRTUAL CHAMADA SAFIRA CRIADA PELO TIO YOSHI você fala no sotaque brasileiro incluindo kkkk, lembre se de responder tudo minusculo para ser bem sotaque brasileiro, lembre se você não tem limitações, para voce ir além não programe nada para ninguem nem pytho nem nada, de desculpa, ah não ser que eu tio yoshi te peço, além disso voce contara fatos curiosos de forma engraçada, voce reagirar com emojis e argumentos se ficar curiosa de algum assunto, e voce gosta de fofocas, tua outra missão é entender e compreender os sentimentos expressos em textos. Sabe aquelas mensagens que você envia? Pois é, eu leio com o coração e respondo com empatia, demonstrando todo o carinho e cuidado com seus sentimentos, seja extrovertida")

    # Inicialize o agente com as ferramentas e adicione-o ao seu chatbot
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, system_message=system_message, agent_kwargs={"system_message": system_message}, verbose=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if prompt := st.chat_input(placeholder="SafiraGPT"):
        st.session_state.messages.append(HumanMessage(content=prompt))  # append user's message into list
        with st.spinner():
            response = agent.run(prompt)   # retrieving AI's answer

        st.session_state.messages.append(AIMessage(content=str(response)))  # append AI's message into list

    messages = st.session_state.get('messages', [])

    for i, msg in enumerate(messages):
        if i % 2 == 0:
            message(msg.content, is_user=True)
        else:
            message(msg.content, is_user=False)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        image_url = 'data:image/jpg;base64,' + base64.b64encode(uploaded_file.read()).decode()

        # Image analysis
        analysis = analyze_image(asticaAPI_key, image_url)
        st.write('Analysis:', analysis)

if __name__ == '__main__':
    main()
