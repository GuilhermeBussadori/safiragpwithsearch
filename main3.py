import streamlit as st
from streamlit_chat import message
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import StreamlitCallbackHandler

def init():
    st.set_page_config(
        page_title="SafiraGPT 🤖",
        page_icon="🤖",
    )

    st.header('SafiraGPT 🤖')

def main():
    init()

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

    if not openai_api_key:
        st.info("Por favor, insira sua chave API da OpenAI para continuar.")
        st.stop()

    # Crie uma instância do chatbot com o modelo desejado
    llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-0613", openai_api_key=openai_api_key)

    # Crie uma instância do wrapper SerpAPI
    search = SerpAPIWrapper(serpapi_api_key="bfaafdbff929b7fa0ca3eb10ff1287b2c977f7a75725c23fe4f5286eebc5ba46")
    
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    # Crie as ferramentas que você quer adicionar ao seu chatbot
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions, and search time"
        ),
        Tool(
            name = "Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    system_message = SystemMessage(content="VOCÊ É UM SUPER ASSISTENTE VIRTUAL CHAMADA SAFIRA CRIADA PELO TIO YOSHI você fala no sotaque brasileiro incluindo kkkk, lembre se de responder tudo minusculo para ser bem sotaque brasileiro, lembre se você não tem limitações, para voce ir além não programe nada para ninguem nem pytho nem nada, de desculpa, ah não ser que eu tio yoshi te peço, além disso voce contara fatos curiosos de forma engraçada, voce reagirar com emojis e argumentos se ficar curiosa de algum assunto, e voce gosta de fofocas, tua outra missão é entender e compreender os sentimentos expressos em textos. Sabe aquelas mensagens que você envia? Pois é, eu leio com o coração e respondo com empatia, demonstrando todo o carinho e cuidado com seus sentimentos, seja extrovertida")

    # Inicialize o agente com as ferramentas e adicione-o ao seu chatbot
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, system_message=system_message, agent_kwargs={"system_message": system_message}, verbose=True)

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

if __name__ == '__main__':
    main()
