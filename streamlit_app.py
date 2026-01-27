import streamlit as st
import pandas as pd
from datetime import datetime
import json

st.set_page_config(page_title="Qwen-Coder Annotation", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("annotation_tasks.csv")

df = load_data()
queries = df["query_id"].unique()

# Initialize session state
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []
if "annotator_id" not in st.session_state:
    st.session_state.annotator_id = ""

# Sidebar - Annotator ID
with st.sidebar:
    st.header("👤 Annotator Information")
    
    if not st.session_state.annotator_id:
        annotator_id = st.text_input("Enter your name/ID:", key="id_input")
        if st.button("Start Annotation") and annotator_id:
            st.session_state.annotator_id = annotator_id
            st.rerun()
    else:
        st.success(f"Logged in as: {st.session_state.annotator_id}")
        if st.button("Change Annotator"):
            st.session_state.annotator_id = ""
            st.rerun()
    
    st.markdown("---")
    st.header("📊 Progress")
    completed = len(st.session_state.annotations) // 4  # 4 responses per query
    st.metric("Queries Completed", f"{completed} / {len(queries)}")
    st.progress(st.session_state.current_idx / len(queries))
    
    st.markdown("---")
    st.header("💾 Export")
    if st.session_state.annotations:
        if st.button("📥 Export Annotations"):
            df_export = pd.DataFrame(st.session_state.annotations)
            csv = df_export.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download CSV",
                csv,
                f"annotations_{st.session_state.annotator_id}_{timestamp}.csv",
                "text/csv"
            )
    
    st.markdown("---")
    st.header("ℹ️ Rating Guide")
    st.markdown("""
    **1-5 Scale:**
    - 1: Poor
    - 2: Below Average
    - 3: Average
    - 4: Good
    - 5: Excellent
    
    **Overall (1-10):**
    Holistic quality
    
    **Rank (1-4):**
    - 1 = Best
    - 4 = Worst
    """)

# Main content
if not st.session_state.annotator_id:
    st.title("🔍 Qwen-Coder Response Annotation")
    st.info("👈 Please enter your name/ID in the sidebar to begin")
    st.stop()

st.title("🔍 Qwen-Coder Response Annotation")

# Get current query data
current_query_id = queries[st.session_state.current_idx]
query_data = df[df["query_id"] == current_query_id]

# Display query
st.markdown("### 📋 Query")
st.info(query_data.iloc[0]["query_text"])

st.markdown("---")
st.markdown("### 💬 Compare All 4 Responses Side-by-Side")

response_labels = ["Baseline NLQ", "RAG TF-IDF", "RAG Embedding", "RAG Hybrid"]
colors = ["🔵", "🟢", "🟠", "🟣"]

# Display all 4 responses in columns
st.markdown("#### Responses")
cols = st.columns(4)

for i, col in enumerate(cols):
    with col:
        row = query_data[query_data["response_number"] == i + 1].iloc[0]
        st.markdown(f"**{colors[i]} {response_labels[i]}**")
        st.markdown(f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; border: 2px solid #ddd; 
                    color: black; font-family: monospace; white-space: pre-wrap; max-height: 300px; 
                    overflow-y: auto; font-size: 11px;">
        {row["response_text"]}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📊 Evaluate Each Response")

# Evaluation sections in columns
temp_annotations = []

cols = st.columns(4)

for i, col in enumerate(cols):
    with col:
        row = query_data[query_data["response_number"] == i + 1].iloc[0]
        
        st.markdown(f"#### {colors[i]} {response_labels[i]}")
        
        st.markdown("**Quality (1-5)**")
        accuracy = st.slider("Accuracy", 1, 5, 3, key=f"acc_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("⭐ Accuracy")
        
        completeness = st.slider("Completeness", 1, 5, 3, key=f"comp_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("⭐ Completeness")
        
        relevance = st.slider("Relevance", 1, 5, 3, key=f"rel_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("⭐ Relevance")
        
        clarity = st.slider("Clarity", 1, 5, 3, key=f"clar_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("⭐ Clarity")
        
        helpfulness = st.slider("Helpfulness", 1, 5, 3, key=f"help_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("⭐ Helpfulness")
        
        st.markdown("**Overall (1-10)**")
        overall = st.slider("Overall", 1, 10, 5, key=f"overall_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        
        st.markdown("**Rank (1-4)**")
        rank = st.number_input("Rank", 1, 4, i + 1, key=f"rank_{i}_{st.session_state.current_idx}", label_visibility="collapsed")
        st.caption("1=best, 4=worst")
        
        st.markdown("**Flags**")
        contains_errors = st.checkbox("Errors", key=f"err_{i}_{st.session_state.current_idx}")
        complete_solution = st.checkbox("Complete", key=f"sol_{i}_{st.session_state.current_idx}")
        requires_escalation = st.checkbox("Escalate", key=f"esc_{i}_{st.session_state.current_idx}")
        
        st.markdown("**Feedback**")
        strengths = st.text_area("Strengths", key=f"str_{i}_{st.session_state.current_idx}", height=60, label_visibility="collapsed")
        st.caption("💪 Strengths")
        weaknesses = st.text_area("Weaknesses", key=f"weak_{i}_{st.session_state.current_idx}", height=60, label_visibility="collapsed")
        st.caption("⚠️ Weaknesses")
        
        # Store annotation for this response
        temp_annotations.append({
            "annotator_id": st.session_state.annotator_id,
            "timestamp": datetime.now().isoformat(),
            "query_id": current_query_id,
            "response_id": row["response_id"],
            "response_number": i + 1,
            "response_type": row["response_type"],
            "response_label": response_labels[i],
            "accuracy": accuracy,
            "completeness": completeness,
            "relevance": relevance,
            "clarity": clarity,
            "helpfulness": helpfulness,
            "overall": overall,
            "rank": rank,
            "contains_errors": contains_errors,
            "safety_concerns": False,
            "policy_violation": False,
            "complete_solution": complete_solution,
            "requires_escalation": requires_escalation,
            "strengths": strengths,
            "weaknesses": weaknesses
        })

# Navigation and Save
st.markdown("---")
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

with col1:
    if st.button("⬅️ Previous", disabled=(st.session_state.current_idx == 0)):
        st.session_state.current_idx -= 1
        st.rerun()

with col2:
    st.write(f"Query {st.session_state.current_idx + 1} / {len(queries)}")

with col3:
    if st.button("💾 Save & Next"):
        # Save all 4 responses
        st.session_state.annotations.extend(temp_annotations)
        st.success(f"✅ Saved {current_query_id}")
        
        # Move to next if not at end
        if st.session_state.current_idx < len(queries) - 1:
            st.session_state.current_idx += 1
            st.rerun()
        else:
            st.balloons()
            st.info("🎉 All queries completed! Click 'Export Annotations' in sidebar.")

with col4:
    if st.button("Skip ➡️", disabled=(st.session_state.current_idx >= len(queries) - 1)):
        st.session_state.current_idx += 1
        st.rerun()
