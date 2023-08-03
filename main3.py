import streamlit as st
from streamlit_chat import message
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import StreamlitCallbackHandler
import requests
from transformers import Blip2Processor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from tempfile import NamedTemporaryFile
from io import BytesIO

def generate_image(description, openai_api_key):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    data = {
        "prompts": [description],
        "max_tokens": 512
    }

    response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=data)
    image_url = response.json()['image']['url']

    return image_url

def get_image_caption(image):
    model_name = "Salesforce/blip-image-captioning-large"
    device = "cpu"  # cuda

    processor = Blip2Processor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

def detect_objects(image):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += ' {}'.format(model.config.id2label[int(label)])
        detections += ' {}\n'.format(float(score))

    return detections

def init():
    st.set_page_config(
        page_title="SafiraGPT 🤖",
        page_icon="🤖",
    )

    st.header('SafiraGPT 🤖')

# ... código anterior ...

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
        ),
        Tool(
            name = "ImageCaption",
            func=lambda image_path: get_image_caption(image_path),
            description="useful for when you need to generate a caption for an image"
        ),
        Tool(
            name = "ObjectDetection",
            func=lambda image_path: detect_objects(image_path),
            description="useful for when you need to detect objects in an image"
        ),
        Tool(
            name = "ImageGeneration",
            func=lambda description: generate_image(description, openai_api_key),
            description="useful for when you need to generate an image from a description"
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

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Image caption
        caption_response = agent.run("ImageCaption", image=image)

        # Object detection
        detection_response = agent.run("ObjectDetection", image=image)

        st.write('Caption:', caption_response)
        st.write('Detected objects:', detection_response)

if __name__ == '__main__':
    main()
