# ui.py
import streamlit as st
import requests

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Gemma RAG PDF Assistant",
    page_icon="🤖",
    layout="centered"
)

FASTAPI_URL = "http://127.0.0.1:8000"

st.title("🤖 Gemma 4 Cloud - PDF RAG Assistant")
st.write("Upload a PDF document, index its contents, and ask context-grounded questions.")

# Create two distinct visual sections using structural tabs
tab1, tab2 = st.tabs(["📤 Upload & Ingestion", "💬 Document Chat"])

# --- TAB 1: FILE INGESTION LAYER ---
with tab1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Document", type="primary"):
            with st.spinner("Extracting text and generating local vector embeddings..."):
                try:
                    # Multi-part form data binding matching FastAPI expectations
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    
                    response = requests.post(f"{FASTAPI_URL}/upload_pdf/", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"🎉 Success! Document parsed into {data['num_chunks']} semantic chunks.")
                        st.info(f"Document Registered ID: {data['message'].split(': ')[1]}")
                        
                        # --- NEW BLOCK: RENDER EXTRACTED CHUNKS ---
                        # Use an expander to keep the UI clean while allowing deep data inspection
                        with st.expander("👀 View Extracted Document Chunks", expanded=False):
                            extracted_chunks = data.get("chunks", [])
                            for i, chunk in enumerate(extracted_chunks):
                                # Display index and character count for analytical auditing
                                st.markdown(f"**📦 Chunk {i}** `[{len(chunk)} characters]`")
                                # Use st.info or st.code to create a nice visual bounding box around the text
                                st.info(chunk)
                                st.divider() # Adds a subtle horizontal line between chunks
                                
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to FastAPI backend: {str(e)}")# --- TAB 2: RETRIEVAL & GENERATION INTERACTIVE CHAT ---
with tab2:
    st.header("Chat with your PDF")
    
    # Initialize stateful message history array inside Streamlit session storage memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display prior conversational history items across re-renders
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capturing reactive user prompt input box
    if user_question := st.chat_input("Ask something about the indexed document..."):
        # Append user text to active visual display state
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Trigger response block generation matching assistant state interface
        with st.chat_message("assistant"):
            with st.spinner("Searching vectors & generating grounded response..."):
                try:
                    # Construct Pydantic-compliant JSON model payload string
                    payload = {"question": user_question}
                    response = requests.post(f"{FASTAPI_URL}/query_pdf/", json=payload)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        answer = response_data["answer"]
                        sources = response_data.get("Sources_Referred", "Unknown")
                        
                        # Format output layout safely using markdown formatting
                        formatted_response = f"{answer}\n\n*📊 Sources Referred: {sources} chunk(s)*"
                        st.markdown(formatted_response)
                        
                        # Save assistant response payload to long term session matrix
                        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    else:
                        st.error(f"Generation Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Connection failure to API Gateway: {str(e)}")