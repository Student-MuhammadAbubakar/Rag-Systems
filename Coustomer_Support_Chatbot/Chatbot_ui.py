
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(
    page_title="SupportAI — Customer Support Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* ── Base ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.82rem;
    color: #34d399;
    margin-top: 0.5rem;
}
.status-dot {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Chat Container ── */
.chat-container {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

/* ── Message Bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.8rem 0;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin: 0.8rem 0;
    gap: 10px;
    align-items: flex-start;
}
.bot-avatar {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 4px;
}
.msg-bot .bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    padding: 0.75rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 70%;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* ── Suggested Questions ── */
.suggest-label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* ── Sidebar ── */
.sidebar-stat {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    text-align: center;
}
.sidebar-stat .number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}
.sidebar-stat .label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Category Tags ── */
.cat-tag {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.25);
    color: #a78bfa;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin: 2px;
}

/* ── Input ── */
.stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ── Spinner ── */
.thinking {
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 0.6rem 1rem;
}
.thinking span {
    width: 8px;
    height: 8px;
    background: #60a5fa;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
    40% { transform: scale(1.1); opacity: 1; }
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_db():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "faiss_index",
        index_name="cs_support",
        # Changed from hr_policies → cs_support
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3, 
    )
def get_answer(question, chat_history_list):
    """
    Same logic as HR Policy chatbot but tuned for
    customer support tone — friendly, helpful, concise.
    """

  
    history = []
    for human, ai in chat_history_list:
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))

    db  = load_db()
    llm = load_llm()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest customer question, "
         "rewrite it as a clear standalone question. "
         "Do NOT answer it — just rewrite it clearly."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    if history:
        standalone_question = condense_chain.invoke({
            "input": question,
            "chat_history": history
        })
    else:
        standalone_question = question

    docs = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)
   
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly and helpful customer support assistant. "
         "Your name is SupportAI. "
         "Use the context below to answer the customer's question. "
         "Be warm, empathetic, and concise. "
         "If you don't know the answer from the context, "
         "say: I'm sorry, I don't have that information. "
         "Please contact our support team directly for help. "
         "Keep answers to 3-4 sentences maximum. "
         "Always end with a helpful closing like: "
         "Is there anything else I can help you with?\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


    qa_chain  = qa_prompt | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "input"       : question,
        "chat_history": history,
        "context"     : context
    })

    return answer


# ── SIDEBAR ──────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem;'>💬</div>
            <div style='font-weight:700; font-size:1.1rem;
                        color:#a78bfa; margin-top:0.3rem;'>
                SupportAI
            </div>
            <div style='color:#64748b; font-size:0.8rem;'>
                Customer Support Assistant
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Live stats from session state
        total_msgs = len(st.session_state.get("messages", []))
        questions  = len(st.session_state.get("chat_history", []))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='sidebar-stat'>
                <div class='number'>{questions}</div>
                <div class='label'>Questions</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='sidebar-stat'>
                <div class='number'>{total_msgs}</div>
                <div class='label'>Messages</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")


        st.markdown("<div style='color:#64748b; font-size:0.78rem;"
                    "text-transform:uppercase; letter-spacing:0.08em;"
                    "margin-bottom:0.5rem;'>I can help with</div>",
                    unsafe_allow_html=True)

        categories = [
            "📦 Orders", "💳 Payments", "🚚 Shipping",
            "↩️ Returns", "🔑 Account", "🎫 Refunds",
            "📋 Policies", "❓ General"
        ]
        tags_html = "".join(
            f"<span class='cat-tag'>{c}</span>" for c in categories
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("---")


        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()
        

        st.markdown("---")

        st.markdown("""
        <div style='color:#475569; font-size:0.75rem; line-height:1.8;'>
            <div style='color:#64748b; font-weight:600;
                        margin-bottom:0.4rem;'>Built with</div>
            🦙 Groq llama-3.1-8b-instant<br>
            🤗 HuggingFace Embeddings<br>
            📊 FAISS Vector Database<br>
            🔗 LangChain Core<br>
            🚀 Streamlit
        </div>
        """, unsafe_allow_html=True)



SUGGESTED = [
    "How do I cancel my order?",
    "Where is my refund?",
    "My payment was declined, what do I do?",
    "How do I reset my password?",
    "How do I return a product?",
]

def showUI():

    render_sidebar()
    # Render the sidebar first

    # ── Header ──
    st.markdown("""
    <div class='main-header'>
        <h1>SupportAI 💬</h1>
        <p>Your 24/7 intelligent customer support assistant</p>
        <div class='status-badge'>
            <div class='status-dot'></div>
            Online — Ready to help
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages     = []
        st.session_state.chat_history = []

    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class='chat-container'>
            <div class='msg-bot'>
                <div class='bot-avatar'>🤖</div>
                <div class='bubble'>
                    👋 Hi there! I'm <strong>SupportAI</strong>,
                    your customer support assistant.<br><br>
                    I can help you with orders, payments, shipping,
                    returns, refunds, and account issues.<br><br>
                    What can I help you with today?
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            "<div class='suggest-label'>Try asking</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(len(SUGGESTED))
        for i, suggestion in enumerate(SUGGESTED):
            with cols[i]:
                if st.button(suggestion, key=f"suggest_{i}",
                             use_container_width=True):
                    st.session_state._pending_question = suggestion
                    st.rerun()
    
    pending = st.session_state.pop("_pending_question", None)
    if st.session_state.messages:
        chat_html = "<div class='chat-container'>"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f"""
                <div class='msg-user'>
                    <div class='bubble'>{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class='msg-bot'>
                    <div class='bot-avatar'>🤖</div>
                    <div class='bubble'>{msg['content']}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
    user_input = st.chat_input(
        "Type your question here... e.g. Where is my order?"
    )


    question = user_input or pending

    if question:

        st.session_state.messages.append({
            "role"   : "user",
            "content": question
        })
        with st.spinner("SupportAI is thinking..."):
            try:
                answer = get_answer(
                    question=question,
                    chat_history_list=st.session_state.chat_history
                )
            except Exception as e:
                answer = (
                    f"I'm sorry, I encountered an error: {str(e)}. "
                    f"Please try again or contact our support team."
                )

        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer
        })
        st.session_state.chat_history.append((question, answer))

        st.rerun()
if __name__ == "__main__":
    showUI()