import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent,agent_type
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
groq_api_key=os.getenv("GROQ_API_KEY")
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)
search=DuckDuckGoSearchRun(name="search")
st.title("LangChain -Chat with search")
"""
in this example,we are using "StreamlitCallbackHandler" to display the thoughts of actions
Try more langchain,streamlit agent examples on github

"""
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant,"content":"Hi,i'm a chatbot who can search the web.How can i help you?"}

    ]
for msg in st.session.messages:
    st.chat_message(msg["role"].write(msg["content"]))
if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
    llm=ChatGroq(api_key=groq_api_key,model="llama-3.1-8b-instant",streaming=True)
    tools=[search,arxiv,wiki]
    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)

        













