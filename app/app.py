import streamlit as st
from provider import GeminiProvider
from pathlib import Path
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

file_path_icon = Path(__file__).parent.parent / "Logos32.32px.png"
st.set_page_config(page_title="Business Chatbot", page_icon=file_path_icon, layout="wide")
file_path_logo = Path(__file__).parent.parent / "poweredbymatedata.png"
st.image(file_path_logo, width=400)

client = GeminiProvider(api_key=os.getenv("GEMINI_API_KEY"), model=os.getenv("GEMINI_MODEL"))

with st.sidebar:
    st.title("⚙️ Settings")
 
    model = st.selectbox(
        "Model",
        options=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ],
        index=0,
    )
 
    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise assistant.",
        height=100,
    )
 
    max_tokens = st.slider("Max output tokens", 256, 4096, 1024, step=256)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.1)
 
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
 
st.title("💬 Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Render historical chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        async def stream_wrapper():
            async for chunk in client.astream(
                prompt, system=system_prompt, temperature=temperature
            ):
                yield chunk

        def sync_generator():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agen = stream_wrapper()
            try:
                while True:
                    yield loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                pass
            finally:
                loop.close()
        full_response = st.write_stream(sync_generator())

    # Save complete response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})