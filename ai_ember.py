import pickle5 as pk
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.agents import AgentType, load_tools, initialize_agent, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import tool
import openai
import streamlit as st
import embeddings as emb
from pydantic import BaseModel, Field
import time
import pandas as pd

fire_risk = pd.read_csv('assets/data/fire_risk.csv')

print(fire_risk)

# Check if API key is stored in session state, if not, get it from secrets.
if openai.api_key not in st.session_state:
  openai.api_key = st.secrets["OPENAI_API_KEY"]

fire_risk_db = emb.loadFireEmbeddings()
preperation_db = emb.loadPreperationsEmbeddings()


# Open and read the 'header.md' file, and store its content in the 'body' variable.
with open('assets/copy/header.md', 'r') as f:
    body = f.read()

# Display header icon
st.image('assets/images/Logo.png', width=150)

# Render the content of 'body' as Markdown.
st.markdown(body, unsafe_allow_html=False, help=None)


@tool("fireRiskInfo", return_direct=True)
def retrieve_fire_risk_info(query:str)->str:
    """useful for when you need to retrieve the fire risk information for a given location or type of location"""
    similar_response = fire_risk_db.similarity_search(query, k=1)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

def retrieve_preperation_info(query):
    similar_response = preperation_db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array


with open('assets/prompts/InitialPrompt.txt', 'r') as f:
    initial_system_prompt = f.read()
    f.close()
                                                   
welcomePrompt = "Hey :)"

# Create a StreamHandler class to handle the streaming of new tokens to give the effect of typing when the LLM is responding
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if 'action' in kwargs:
            if kwargs['action'] == 'Final Answer':
                self.text += kwargs['action_input']
                self.container.markdown(self.text)
        else:
            self.text += token
            self.container.markdown(self.text)


# Initialize chat messages if not already present in the session state.
#if "messages" not in st.session_state:
  # Load initial system messages into the chat.
  #st.session_state["messages"] = [ChatMessage(role="system", content=initial_system_prompt)]

if "setup_state" not in st.session_state:
  st.session_state["setup_state"]=True

st.chat_message("assistant").write("Let's start by picking the environment that best describes your house?")
if(st.session_state["setup_state"]==True):time.sleep(0.8)
if st.button("Close to or among grass or paddocks",key="grass", type="secondary"):
    st.write('Why hello there')
    st.session_state["property_type"]="grass"
    situational_info = fire_risk.query("Environment.str.contains('Close to or among grass or paddocks', case=False)") 
    st.session_state["what_to_expect"]=situational_info['What can you expect'].values
    st.session_state["what_to_do"]=situational_info["What to do"].values
    print(st.session_state["what_to_expect"])
    print(st.session_state["what_to_do"])
    st.session_state["setup_state"]=False
if(st.session_state["setup_state"]==True):time.sleep(0.2)
if st.button("Close to or among dense or open bush",key="bush",  type="secondary"):
   st.session_state["property_type"]="bush"
   st.session_state["setup_state"]=False
if(st.session_state["setup_state"]==True):time.sleep(0.2)
if st.button("Near coastal scrub",key="coastal",  type="secondary"):
    st.session_state["property_type"]="coastal"
    st.session_state["setup_state"]=False
if(st.session_state["setup_state"]==True):time.sleep(0.2)
if st.button("Where cities and towns meet the bush",key="cities",  type="secondary"):
    st.session_state["property_type"]="cities"
    st.session_state["setup_state"]=False
if(st.session_state["setup_state"]==True):time.sleep(0.2)
if st.button("Where cities and towns meet grasslands",key="urban_frindge",  type="secondary"):
    st.session_state["property_type"]="urban_frindge"
    st.session_state["setup_state"]=False
if(st.session_state["setup_state"]==True):time.sleep(0.2)
if st.button("Suburban or Urban",key="urban",  type="secondary"):
    st.session_state["property_type"]="urban"
    st.session_state["setup_state"]=False




  # Load assistant’s welcome messages into the chat.
  #st.session_state.messages.append(ChatMessage(role="assistant", content=""))
  #st.session_state.messages.append(ChatMessage(role="assistant", content=welcomePrompt))
  #st.session_state.messages.append(ChatMessage(role="assistant", content="Ready to start?"))


if st.session_state["setup_state"]==False:
  st.session_state["messages"] = [ChatMessage(role="system", content=initial_system_prompt)]
  st.session_state["messages"].append(ChatMessage(role="system", content="The user lives in a " + str(st.session_state["property_type"]) + " environment"))
  st.session_state["messages"].append(ChatMessage(role="system", content="They should expect to see " + str(st.session_state["what_to_expect"])))
  st.session_state["messages"].append(ChatMessage(role="system", content="They should do the following " + str(st.session_state["what_to_do"])))
  st.session_state["messages"].append(ChatMessage(role="assistant", content="Thanks for letting me know that. Are you ready to chat?"))
  #st.session_state.messages.append(ChatMessage(role="assistant", content="Thanks for letting me know that. Are you ready to chat?"))
  # Display existing chat messages.
  for msg in st.session_state.messages:
      #if (msg.role == "assistant" or msg.role == "user"):
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
                          callbacks=[stream_handler],
                          verbose=False)
          
          #tools = Tool.from_function(func=retrieve_fire_risk_info.run, 
          #                          name="Fire Risk Information Gathering Tool",
          #                          description="useful for when you need to retrieve the fire risk information for a given location or type of location")
          
          #agent_kwargs = {
          #    "extra_prompt_messages": [ConversationBufferWindowMemory(memory_key="chat_history")],
          #}

          #agent = initialize_agent([tools], 
          #                        llm, 
          #                        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
          #                        handle_parsing_errors=True,
          #                        agent_kwargs=agent_kwargs)
          

        
          # Generate and display the assistant's response
          response = llm(st.session_state.messages)
          st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))

          
          # Generate and display the assistant's response
          
          #response = agent({'chat_history': st.session_state.messages, 'input': prompt})
          #st.session_state.messages.append(ChatMessage(role="assistant", content=response['output']))
