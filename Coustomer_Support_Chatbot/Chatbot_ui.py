# ============================================================
# FILE     : chatbot_ui.py
# PROJECT  : Customer Support Chatbot
# PURPOSE  : Streamlit UI — engaging, portfolio-ready
#            FIXED: LLM now answers ONLY from FAISS dataset
# RUN      : streamlit run chatbot_ui.py
# ============================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
# Loads GROQ_API_KEY from .env file


# ── PAGE CONFIG ─────────────────────────────────────────────

st.set_page_config(
    page_title="SupportAI — Customer Support Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── CUSTOM CSS ──────────────────────────────────────────────

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
    50%       { opacity: 0.4; }
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

/* ── User Bubble ── */
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

/* ── Bot Bubble ── */
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

/* ── Suggested Questions Label ── */
.suggest-label {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1.5rem;
}

/* ── Sidebar Stats ── */
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

/* ── Chat Input ── */
.stChatInput textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: white !important;
}

/* ── Hide Streamlit Defaults ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

</style>
""", unsafe_allow_html=True)


# ── FAISS INDEX CHECK ────────────────────────────────────────

if not os.path.exists("faiss_index/cs_support.faiss"):
    st.error(
        "FAISS index not found! "
        "Please run: python upload_vectors.py first"
    )
    st.stop()
# If FAISS index does not exist stop the app immediately
# Show a clear error instead of a confusing crash message


# ── CACHE: LOAD DB AND LLM ONCE PER SESSION ─────────────────

@st.cache_resource
def load_db():
    # Loads FAISS index ONCE and caches it
    # Without caching it reloads on every single message
    # which adds 10-15 seconds per query
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
        "faiss_index",
        index_name="cs_support",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def load_llm():
    # Loads Groq LLM ONCE and caches it
    # temperature=0: no creativity — sticks strictly to context
    # This is the KEY FIX for LLM answering from own knowledge
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        # temperature=0 = strictly factual from retrieved context
        # temperature=0.3 was allowing LLM to "improvise" answers
        # from its own training data instead of your dataset
    )


# ── CORE: GET ANSWER ─────────────────────────────────────────

def get_answer(question, chat_history_list):
    """
    FIXED VERSION:
    - Strict system prompt forces LLM to use ONLY dataset
    - k=6 retrieves more chunks for better coverage
    - Context verification before calling LLM
    - temperature=0 prevents LLM from using own knowledge
    """

    # Step 1: Convert chat history list to LangChain messages
    history = []
    for human, ai in chat_history_list:
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))
    # LangChain needs HumanMessage and AIMessage objects
    # not plain Python strings

    db        = load_db()
    llm       = load_llm()
    retriever = db.as_retriever(search_kwargs={"k": 6})
    # k=6: retrieve top 6 most relevant chunks
    # Increased from k=4 to give LLM more context to work with

    # Step 2: Rewrite question using chat history
    # So "what about refunds?" becomes "what is the refund policy?"
    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest customer question, "
         "rewrite it as a clear standalone question. "
         "Do NOT answer it. Just rewrite it clearly."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    condense_chain = condense_prompt | llm | StrOutputParser()

    if history:
        standalone_question = condense_chain.invoke({
            "input"       : question,
            "chat_history": history
        })
    else:
        standalone_question = question
    # If no chat history yet use original question directly

    # Step 3: Retrieve relevant chunks from FAISS vector database
    docs    = retriever.invoke(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)
    # Join all 6 retrieved chunks into one context string
    # This context is what the LLM MUST use to answer

    # Step 4: Verify context was actually retrieved
    if not context.strip():
        return (
            "I'm sorry, I could not find relevant information "
            "in our knowledge base for your question. "
            "Please contact our support team directly for help. "
            "Is there anything else I can help you with?"
        )
    # If FAISS returned empty context do not call LLM at all
    # Return a polite fallback message immediately
    # This prevents LLM from making up an answer

    # Step 5: STRICT QA prompt — KEY FIX
    # This is the most important change
    # The old prompt was too weak and LLM was ignoring context
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a customer support assistant named SupportAI.\n\n"

         "STRICT RULES — YOU MUST FOLLOW ALL OF THESE:\n"
         "1. Answer ONLY using the CONTEXT provided below.\n"
         "2. Do NOT use your own training knowledge or memory.\n"
         "3. Do NOT make up any information not in the context.\n"
         "4. If the answer is not in the context say exactly:\n"
         "   I don't have that information in our knowledge base.\n"
         "   Please contact our support team directly for help.\n"
         "5. Keep answers to 3 to 4 sentences maximum.\n"
         "6. Be warm, empathetic, and friendly in tone.\n"
         "7. Always end with: Is there anything else I can help you with?\n\n"

         "CONTEXT FROM KNOWLEDGE BASE:\n"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
         "{context}\n"
         "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

         "Remember: Answer ONLY from the context above. "
         "Never use outside knowledge."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    # The STRICT RULES section is the key fix
    # Old prompt just said "use context below"
    # New prompt explicitly forbids using own knowledge
    # and tells LLM exactly what to say when it does not know

    qa_chain = qa_prompt | llm | StrOutputParser()
    # Pipe operator: prompt → llm → parse as string

    answer = qa_chain.invoke({
        "input"       : standalone_question,
        "chat_history": history,
        "context"     : context
    })
    # standalone_question is used instead of original question
    # because it is the rewritten clear version

    return answer


# ── SIDEBAR ──────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:

        # Bot identity
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

        # Live stats
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

        # Category tags showing what bot can help with
        st.markdown(
            "<div style='color:#64748b; font-size:0.78rem;"
            "text-transform:uppercase; letter-spacing:0.08em;"
            "margin-bottom:0.5rem;'>I can help with</div>",
            unsafe_allow_html=True
        )
        categories = [
            "📦 Orders", "💳 Payments", "🚚 Shipping",
            "↩️ Returns", "🔑 Account", "🎫 Refunds",
            "📋 Policies", "❓ General"
        ]
        tags_html = "".join(
            f"<span class='cat-tag'>{c}</span>"
            for c in categories
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("---")

        # Clear conversation button
        if st.button("🗑️ Clear Conversation",
                     use_container_width=True):
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()
        # Clears both display messages and LangChain history
        # st.rerun() refreshes page so chat window empties immediately

        st.markdown("---")

        # Tech stack for portfolio viewers
        st.markdown("""
        <div style='color:#475569; font-size:0.75rem; line-height:1.8;'>
            <div style='color:#64748b; font-weight:600;
                        margin-bottom:0.4rem;'>Built with</div>
            🦙 Groq llama-3.1-8b-instant<br>
            🤗 HuggingFace all-MiniLM-L6-v2<br>
            📊 FAISS Vector Database<br>
            🔗 LangChain Core<br>
            🚀 Streamlit<br>
            📂 Bitext Customer Support Dataset
        </div>
        """, unsafe_allow_html=True)


# ── SUGGESTED QUESTIONS ──────────────────────────────────────

SUGGESTED = [
    "How do I cancel my order?",
    "Where is my refund?",
    "My payment was declined",
    "How do I reset my password?",
    "How do I return a product?",
]
# Clickable buttons shown when chat is empty
# Helps users instantly know what to ask
# Very important for portfolio demonstrations


# ── MAIN UI ──────────────────────────────────────────────────

def showUI():

    render_sidebar()

    # ── Header ──
    st.markdown("""
    <div class='main-header'>
        <h1>SupportAI 💬</h1>
        <p>Your 24/7 intelligent customer support assistant</p>
        <div class='status-badge'>
            <div class='status-dot'></div>
            Online — Answering from knowledge base only
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Status badge updated to say "Answering from knowledge base only"
    # So users and portfolio viewers know RAG is working correctly

    # ── Session State Init ──
    if "messages" not in st.session_state:
        st.session_state.messages     = []
        st.session_state.chat_history = []
    # messages     : list of {role, content} dicts for display
    # chat_history : list of (question, answer) tuples for LangChain memory

    # ── Welcome Message when chat is empty ──
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class='chat-container'>
            <div class='msg-bot'>
                <div class='bot-avatar'>🤖</div>
                <div class='bubble'>
                    👋 Hi there! I'm <strong>SupportAI</strong>,
                    your customer support assistant.<br><br>
                    I can help you with <strong>orders, payments,
                    shipping, returns, refunds,</strong> and
                    <strong>account issues</strong>.<br><br>
                    All my answers come directly from our
                    knowledge base. What can I help you with today?
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Welcome message mentions "knowledge base"
        # so users know answers are from dataset not made up

        # ── Suggested Question Buttons ──
        st.markdown(
            "<div class='suggest-label'>Try asking</div>",
            unsafe_allow_html=True
        )
        cols = st.columns(len(SUGGESTED))
        for i, suggestion in enumerate(SUGGESTED):
            with cols[i]:
                if st.button(
                    suggestion,
                    key=f"suggest_{i}",
                    use_container_width=True
                ):
                    st.session_state._pending = suggestion
                    st.rerun()
        # Each button stores the suggestion as pending question
        # st.rerun() triggers the chat to process it

    # ── Handle Suggested Question Click ──
    pending = st.session_state.pop("_pending", None)
    # .pop() gets the pending question AND removes it
    # so it is only processed once not on every rerun

    # ── Display Full Chat History ──
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
    # Renders all messages as styled purple/teal bubbles
    # User messages: purple bubble on the right
    # Bot messages: teal avatar + grey bubble on the left

    # ── Chat Input Box ──
    user_input = st.chat_input(
        "Type your question... e.g. How do I cancel my order?"
    )

    # Use either typed input or clicked suggestion button
    question = user_input or pending

    if question:

        # Immediately add user message to session
        st.session_state.messages.append({
            "role"   : "user",
            "content": question
        })

        # Get answer from RAG pipeline
        with st.spinner("Searching knowledge base..."):
            try:
                answer = get_answer(
                    question=question,
                    chat_history_list=st.session_state.chat_history
                )
            except Exception as e:
                answer = (
                    "I'm sorry, I encountered a technical issue. "
                    f"Details: {str(e)}. "
                    "Please try again or contact our support team."
                )
        # Spinner says "Searching knowledge base" not "Thinking"
        # This makes it clear the bot is searching your dataset

        # Add assistant answer to session
        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer
        })

        # Save to LangChain chat history for conversation memory
        st.session_state.chat_history.append((question, answer))

        st.rerun()
        # Rerun refreshes page to show new messages
        # in the styled chat bubbles above


# ── ENTRY POINT ──────────────────────────────────────────────

if __name__ == "__main__":
    showUI()