import streamlit as st
from rag_pipeline import query_rag

st.set_page_config(page_title="PolicyPilot AI", layout="centered")
st.title("🧠 PolicyPilot AI")
st.markdown("Ask a question about your policy documents (e.g., BOI, AIB, Central Bank)")

query = st.text_input("Enter your question")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        answer, sources = query_rag(query)
        st.success(answer)

        st.markdown("**Sources:**")
        for src in sources:
            st.write(f"- {src}")
