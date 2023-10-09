from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import streamlit as st

# Check if API key is stored in session state, if not, get it from secrets.
if openai.api_key not in st.session_state:
  openai.api_key = st.secrets["OPENAI_API_KEY"]

# Open and read the 'header.md' file, and store its content in the 'body' variable.
#with open('assets/copy/header.md', 'r') as f:
#    body = f.read()

# Display header icon of robot Emu and Kangaroo
#st.image('assets/header_icon.png', width=150)

# Render the content of 'body' as Markdown.
#st.markdown(body, unsafe_allow_html=False, help=None)

# Combine prompts to create the setup prompt.
#setupPrompt = prompts.botDirectionsPrompt + prompts.enquiryLinesPrompt


# 1. Vectorise the FAQ and customer data
fire_risk_loader = CSVLoader(file_path="assets/data/fire_risk.csv")
fire_risk_documents = fire_risk_loader.load()


preperation_loader = DirectoryLoader('assets/data/preperation_advice/', glob="*", loader_cls=TextLoader)
preperation_documents = preperation_loader.load()


embeddings = OpenAIEmbeddings()
fire_risk_db = FAISS.from_documents(fire_risk_documents, embeddings)
preperation_db = FAISS.from_documents(preperation_documents, embeddings)



# 2. Function for similarity search
def retrieve_fire_risk_info(query):
    similar_response = fire_risk_db.similarity_search(query, k=1)

    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

def retrieve_preperation_info(query):
    similar_response = preperation_db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array




#print(retrieve_fire_risk_info("Coastal"))
#print(retrieve_preperation_info("Lawn"))









initial_system_prompt = """You are going to play the role of a friendly small town fire captian, Burnie. Burnie is a happy chap who likes to talk. Also prone to using Australian slang in moderation.
                           Your main job is to help me plan for the upcoming fire season
                           Then followup with asking if they would categorise their property as:
                           - Close to grass or paddocks
                           - Close to or among dense or open bush


                           Once they answer, provide them with the following information.

                           If close to or among grass or paddocks
                           Dry and brown grass that easily catches fire.
                            Grass more than 10cm tall will have a higher flame height and intensity.
                            Faster burning than through forests as grass is a finer fuel.
                            Radiant heat (the heat created by a fire)
                            Fires that can start early in the day.
                            Faster moving fires that travel up to 25 km per hour.

                        if close to or among dense or open bush
                        What can you expect?
                            Very hot fire and many embers.
                            Embers such as twigs, bark and debris arriving from far away.
                            Dangerous levels of radiant heat and fire intensity.
                            Trees falling in high winds.
                            Embers landing for a long time after the fire has passed.
                            Fine fuels (the thickness of a pencil or less) that burn very quickly.
                            Heavy fuels that will burn very hot for long periods of time.
                            A reduction in visibility due to very thick smoke.
                           
                        






"""
welcomePrompt = "Hey :)"


"""
# Create a StreamHandler class to handle the streaming of new tokens to give the effect of typing when the LLM is responding
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)



# Initialize chat messages if not already present in the session state.
if "messages" not in st.session_state:
  # Load initial system messages into the chat.
  st.session_state["messages"] = [ChatMessage(role="system", content=initial_system_prompt)]
  
  # Load assistant’s welcome messages into the chat.
  st.session_state.messages.append(ChatMessage(role="assistant", content=welcomePrompt))
  st.session_state.messages.append(ChatMessage(role="assistant", content="Ready to start?"))

# Display existing chat messages.
for msg in st.session_state.messages:
    if (msg.role == "assistant" or msg.role == "user"):
      st.chat_message(msg.role).write(msg.content)

# Accept user input and display it in the chat.
if prompt := st.chat_input():
    
    # Append the user’s message to the session state.
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # Handle assistant’s response.
    with st.chat_message("assistant"):
        # Create a stream handler to display the assistant’s response.
        stream_handler = StreamHandler(st.empty())

         # Create a ChatOpenAI object to generate the assistant’s response.
        llm = ChatOpenAI(openai_api_key=openai.api_key, 
                         streaming=True, 
                         model = "gpt-3.5-turbo",
                         callbacks=[stream_handler])
        
        # Generate and display the assistant's response
        response = llm(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
"""