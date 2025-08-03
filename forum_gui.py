import streamlit as st
from forum_agent import forum_agent, ask_forum_agent

st.set_page_config(page_title="Bottlehead Forum Agent", page_icon="🤖")
st.title("🤖 Bottlehead Forum Agent")

# Text input for user question
query = st.text_input("Ask a question about Bottlehead kits:", placeholder="e.g., How do I install the Speedball upgrade?")

# Optional product filter
product = st.text_input("Optional: Filter by product name", placeholder="e.g., Crack, BeePre2, Kaiju")

# Display response
if query:
    with st.spinner("🧠 Thinking..."):
        try:
            answer = ask_forum_agent(query, product)
            st.success("✅ Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error answering question: {e}")
