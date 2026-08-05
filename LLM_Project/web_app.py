import streamlit as st
from course_planner import main
from langchain_core.messages import HumanMessage, AIMessage


# Page Configuration
st.set_page_config(
    page_title="SFU SIAT Course Planner",
    page_icon="🎓",
    layout="centered"
)

# Load Existing RAG Chatbot
@st.cache_resource
def load_chatbot():
    return main(web_mode=True)

qa_chain = load_chatbot()

# Session State
# Stores messages shown in the webpage
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stores LangChain conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("About this Assistant")

    st.write(
        """
        This AI-powered course planning assistant uses Retrieval-Augmented Generation
        (RAG) to answer questions based on SFU SIAT course and program information.
        """
    )

    st.write(
        """
        **The assistant can help with:**

        - IAT course prerequisites
        - SIAT degree requirements
        - Academic regulations
        - Course planning
        """
    )

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Header (Main top part of the chat)
st.title("🎓 SFU SIAT Course Planning Assistant")
st.write(
    """Ask questions about SIAT courses, prerequisites, degree requirements, academic regulations, and course planning."""
)
st.caption(
    "Powered locally using Llama 3.1, Ollama, nomic-embed-text, and ChromaDB."
)
st.warning(
    "⚠️ This assistant is for course planning support only. Always verify the latest course requirements, prerequisites, and academic decisions "
    "with an SFU Academic Advisor or the current SFU Academic Calendar."
)

# Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources underneath assistant responses
        if message.get("sources"):
            with st.expander("Sources Referenced"):
                for source in message["sources"]:
                    st.write(f"- {source}")

# User Input
query = st.chat_input(
    "Ask an SIAT course planning question..."
)

# Generate Response
if query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching SFU information..."):
            try:
                response = qa_chain.invoke({
                    "input": query,
                    "chat_history": st.session_state.chat_history
                })

                answer = response["answer"]

                # Get source files retrieved by ChromaDB
                sources = sorted(
                    set(
                        doc.metadata.get(
                            "source",
                            "Unknown File"
                        )
                        for doc in response["context"]
                    )
                )

                # Display answer
                st.markdown(answer)

                # Display sources
                if sources:
                    with st.expander("Sources Referenced"):
                        for source in sources:
                            st.write(f"- {source}")

                # Save assistant response for webpage history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

                # Save conversation for LangChain
                st.session_state.chat_history.append(
                    HumanMessage(content=query)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

            except Exception as error:
                st.error(
                    "The chatbot encountered an error."
                )
                st.code(
                    str(error)
                )