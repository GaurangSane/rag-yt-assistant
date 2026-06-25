import time
import streamlit as st



st.set_page_config(
    page_title = "YouTube RAG Assistant",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


# Enhanced CSS — replace the existing st.markdown CSS block
st.markdown("""
<style>
    /* ── Typography ──────────────────────────────────── */
    .stChatMessage {
        padding      : 0.75rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    /* ── Answer formatting ───────────────────────────── */
    .stChatMessage p {
        line-height: 1.7;
        margin-bottom: 0.5rem;
    }

    /* ── Source bar ──────────────────────────────────── */
    .source-bar {
        display        : flex;
        align-items    : center;
        gap            : 8px;
        margin-top     : 10px;
        padding-top    : 10px;
        border-top     : 1px solid rgba(255,255,255,0.08);
        flex-wrap      : wrap;
    }

    .source-label {
        font-size  : 11px;
        color      : #94a3b8;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ── Metric cards ────────────────────────────────── */
    [data-testid="metric-container"] {
        background   : rgba(255,255,255,0.03);
        border       : 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        padding      : 12px;
    }

    /* ── Chat input ──────────────────────────────────── */
    .stChatInputContainer {
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        padding-top: 12px;
    }

    /* ── Expander polish ─────────────────────────────── */
    .streamlit-expanderHeader {
        font-size  : 12px !important;
        color      : #94a3b8 !important;
    }

    /* ── Success/info boxes ──────────────────────────── */
    .stSuccess, .stInfo {
        border-radius: 8px;
        font-size    : 13px;
    }

    /* ── Status dot animation ────────────────────────── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.5; }
    }
    .pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    
    defaults = {
        "pipeline"            : None,
        "video_url"           : "",
        "video_indexed"       : False,
        "video_chunk_count"   : 0,
        "messages"            : [],
        "conversation_history": [],
        "is_loading"          : False,
        "error_message"       : "",
        "last_query_ms"       : 0.0,
        "last_latency"        : None,   
        "last_queries_used"   : [],     
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default




@st.cache_resource
def load_api_client():
    from src.api_client import RAGApiClient
    return RAGApiClient()  



def render_sidebar() -> None:
    
    with st.sidebar:
        st.title("🎬 YT RAG Assistant")
        st.caption("Chat with any YouTube video")
        st.divider()

       
        with st.expander("⚙️ How it works", expanded=False):
            st.markdown("""
**9-step RAG pipeline:**

1. 📄 Fetch YouTube transcript
2. ✂️  Split into 60s chunks (15s overlap)
3. 🔢 Embed with `all-mpnet-base-v2`
4. 💾 Store in ChromaDB (persistent)
5. 🔄 Rewrite question × 3 variants
6. 🔍 Hybrid search (semantic + BM25 + RRF)
7. 🎯 Rerank with cross-encoder
8. 📝 Build structured prompt
9. 🤖 Generate with LLaMA3 via Groq
            """)

    
        st.subheader("📊 Session Info")

        if st.session_state.video_indexed:
            st.success("✅ Video ready")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Chunks",
                    st.session_state.video_chunk_count,
                    help="Number of 60-second segments indexed"
                )
            with col2:
                st.metric(
                    "Messages",
                    len(st.session_state.messages),
                )

        
            if st.session_state.last_latency:
                lat = st.session_state.last_latency
                st.metric(
                    "Last query",
                    f"{lat.total_ms:.0f}ms",
                    help="Total pipeline time for last question"
                )

                with st.expander("⏱ Latency breakdown", expanded=False):
                    breakdown = {
                        "Query transform": lat.query_transform_ms,
                        "Retrieval"      : lat.retrieval_ms,
                        "Reranking"      : lat.reranking_ms,
                        "Generation"     : lat.generation_ms,
                    }
                    for stage, ms in breakdown.items():
                        
                        bar = "█" * max(1, int(ms // 50))
                        st.caption(f"`{stage}` {ms:.0f}ms")
                        st.caption(f"{bar}")

        
            if st.session_state.last_queries_used:
                with st.expander("🔍 Last search queries", expanded=False):
                    st.caption(
                        "These 3 variants were sent to the vector DB:"
                    )
                    for i, q in enumerate(
                        st.session_state.last_queries_used, 1
                    ):
                        st.caption(f"**{i}.** {q}")

        else:
            st.info("No video loaded yet")

        st.divider()

    
        if st.session_state.messages:
            if st.button(
                "🗑️ Clear conversation",
                use_container_width = True,
                type                = "secondary",
                help                = "Start a fresh conversation on the same video",
            ):
                st.session_state.messages            = []
                st.session_state.conversation_history = []
                st.session_state.last_query_ms       = 0.0
                st.session_state.last_latency        = None
                st.session_state.last_queries_used   = []
                st.rerun()

        if st.session_state.video_indexed:
            if st.button(
                "🔄 Load different video",
                use_container_width = True,
                type                = "secondary",
                help                = "Keep indexed videos, switch to a new one",
            ):
                st.session_state.video_url           = ""
                st.session_state.video_indexed       = False
                st.session_state.video_chunk_count   = 0
                st.session_state.messages            = []
                st.session_state.conversation_history= []
                st.session_state.error_message       = ""
                st.session_state.last_latency        = None
                st.session_state.last_queries_used   = []
                st.rerun()

        
        st.divider()
        st.caption(
            "Built with LangChain · ChromaDB · Groq · Streamlit\n\n"
            "All-mpnet-base-v2 · ms-marco-MiniLM · LLaMA3-8b"
        )



def render_url_input() -> None:
    
    st.subheader("📎 Load a YouTube Video")
    st.caption(
        "Paste any YouTube URL. The video is indexed once — "
        "every question after that responds in under 3 seconds."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        url = st.text_input(
            label            = "YouTube URL",
            placeholder      = "https://www.youtube.com/watch?v=...",
            label_visibility = "collapsed",
            key              = "url_input",
        )
    with col2:
        load_clicked = st.button(
            "🎬 Load",
            type                = "primary",
            use_container_width = True,
        )

    if load_clicked:
        if not url or not url.strip():
            st.warning("⚠️ Please enter a YouTube URL first.")
        else:
            _run_ingestion(url.strip())

    
    st.divider()
    st.caption("🧪 **Example videos to try:**")

    examples = [
        ("3Blue1Brown — Neural Networks",
         "https://www.youtube.com/watch?v=aircAruvnKk"),
        ("Andrej Karpathy — Intro to LLMs",
         "https://www.youtube.com/watch?v=zjkBMFhNj_g"),
    ]
    for title, example_url in examples:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.code(example_url, language=None)
        with col_b:
            st.caption(f"↑ {title}")


def _run_ingestion(url: str) -> None:
    client = st.session_state.pipeline   # now an RAGApiClient

    if client.is_video_indexed(url):
        result = client.ingest(url)   # gets chunk count
        st.session_state.video_url         = url
        st.session_state.video_indexed     = True
        st.session_state.video_chunk_count = result["chunk_count"]
        st.success(f"✅ Already indexed ({result['chunk_count']} chunks)!")
        time.sleep(0.8)
        st.rerun()
        return

    progress = st.progress(0, text="Starting...")
    try:
        with st.status("🔄 Indexing video...", expanded=True) as status:
            st.write("📄 Fetching and processing video...")
            progress.progress(30, text="Ingesting...")

            result = client.ingest(url)

            st.session_state.video_url         = url
            st.session_state.video_indexed     = True
            st.session_state.video_chunk_count = result["chunk_count"]

            progress.progress(100, text="Done!")
            status.update(
                label    = f"✅ Indexed {result['chunk_count']} chunks!",
                state    = "complete",
                expanded = False,
            )

        time.sleep(1.0)
        st.rerun()

    except Exception as e:
        progress.empty()
        _show_ingestion_error(e)

def _show_ingestion_error(error: Exception) -> None:
    
    msg = str(error)

    if "TranscriptNotAvailable" in msg or "disabled" in msg:
        st.error(
            "❌ **No captions available** for this video.\n\n"
            "This video either has captions disabled or is a live stream. "
            "Try a different video — educational content and talks "
            "almost always have captions."
        )
    elif "InvalidYouTubeURL" in msg or "video ID" in msg:
        st.error(
            "❌ **Couldn't parse that URL.**\n\n"
            "Make sure it's a full YouTube URL:\n"
            "`https://www.youtube.com/watch?v=VIDEO_ID`"
        )
    elif "GROQ_API_KEY" in msg:
        st.error(
            "❌ **Groq API key missing.**\n\n"
            "Add `GROQ_API_KEY=your_key` to your `.env` file "
            "and restart the app."
        )
    else:
        st.error(
            f"❌ **Ingestion failed.**\n\n"
            f"Error: `{msg}`\n\n"
            "Check the terminal for the full stack trace."
        )




def render_chat_interface() -> None:
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(
            f"✅ **{st.session_state.video_chunk_count} chunks** "
            f"indexed and ready"
        )
    with col2:
        st.link_button(
            "▶ Open video",
            url                 = st.session_state.video_url,
            use_container_width = True,
        )

    st.divider()

    
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 **Video indexed and ready!**\n\n"
                "I can answer questions about this video's content "
                "and cite the **exact timestamp** for every claim.\n\n"
                "**Try asking:**"
            )
            example_questions = [
                "What is the main topic of this video?",
                "Give me a summary of the key points",
                "What does the speaker say about [topic]?",
                "Explain the most important concept",
            ]
            for q in example_questions:
                if st.button(
                    q,
                    key  = f"example_{q[:20]}",
                    help = "Click to ask this question",
                ):
                    # Trigger the question as if user typed it
                    st.session_state.pending_prompt = q
                    st.rerun()
    
    for message in st.session_state.messages:
        _render_message(message)
    # Handle example question clicked
    if "pending_prompt" in st.session_state:
        prompt = st.session_state.pending_prompt
        del st.session_state.pending_prompt
        _handle_user_question(prompt)

    
    prompt = st.chat_input(
        placeholder = "Ask anything about the video...",
        disabled    = st.session_state.is_loading,
    )

    if prompt and not st.session_state.is_loading:
        _handle_user_question(prompt)


def _render_message(message: dict) -> None:
    """
    Render one message with improved formatting.
    Uses st.markdown for rich text in answers.
    """
    with st.chat_message(message["role"]):

        if message["role"] == "user":
            # User messages — plain text, no markdown needed
            st.write(message["content"])

        else:

        
            st.markdown(message["content"])

            if message.get("answer_grounded") and message.get("sources"):
                _render_sources(message["sources"])

            elif not message.get("answer_grounded", True):
                st.markdown(
                    "<div style='"
                    "font-size:12px;"
                    "color:#64748b;"
                    "margin-top:8px;"
                    "padding:8px 12px;"
                    "background:rgba(255,255,255,0.03);"
                    "border-radius:6px;"
                    "border-left:3px solid #475569"
                    "'>"
                    "ℹ️ This topic isn't covered in the video sections I have access to."
                    "</div>",
                    unsafe_allow_html=True,
                )


def _render_sources(sources: list[dict]) -> None:
    """
    Render sources with visual separator and rank indicators.
    """
    if not sources:
        return

    # Visual separator between answer and sources
    st.markdown(
        "<div style='"
        "border-top:1px solid rgba(255,255,255,0.08);"
        "margin-top:12px;"
        "padding-top:10px;"
        "display:flex;"
        "align-items:center;"
        "gap:8px;"
        "flex-wrap:wrap;"
        "'>"
        "<span style='font-size:11px;color:#64748b;font-weight:500;'>"
        "📍 Sources</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(sources))
    rank_colors = ["#6366f1", "#10b981", "#f59e0b"]   # indigo, green, amber

    for i, (col, source) in enumerate(zip(cols, sources)):
        color = rank_colors[i % len(rank_colors)]
        with col:
            st.link_button(
                label               = f"▶ {source['display']}",
                url                 = source["youtube_link"],
                use_container_width = True,
                help                = (
                    f"Rank {source['rank']} source — "
                    f"opens video at {source['start_time']}"
                ),
            )

def _handle_user_question(prompt: str) -> None:
    client = st.session_state.pipeline   # RAGApiClient

    user_message = {
        "role"           : "user",
        "content"        : prompt,
        "sources"        : [],
        "answer_grounded": True,
    }
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    
    with st.spinner("🤔 Thinking..."):
        try:
                # Convert ConversationTurn objects to dicts for API
            history = st.session_state.conversation_history

            response = client.chat(
                    youtube_url = st.session_state.video_url,
                    question    = prompt,
                    history     = history,
                )
            error = None
        except Exception as e:
            response = None
            error    = str(e)

    if error:
        assistant_message = {
            "role"           : "assistant",
            "content"        : f"⚠️ Error: {error}",
            "sources"        : [],
            "answer_grounded": False,
        }
    else:
        assistant_message = {
            "role"           : "assistant",
            "content"        : response.answer,
            "sources"        : [
                {
                    "rank"        : s.rank,
                    "start_time"  : s.start_time,
                    "end_time"    : s.end_time,
                    "youtube_link": s.youtube_link,
                    "display"     : s.display,
                }
                for s in response.sources
            ],
            "answer_grounded": response.answer_grounded,
        }
        st.session_state.last_query_ms = response.total_ms

    st.session_state.messages.append(assistant_message)
    _render_message(assistant_message)

    if not error and response.answer_grounded:

        st.session_state.conversation_history.append(
            {
                "question" : prompt,
                "answer"   : response.answer,
            }
        )


def render_main() -> None:
    
    st.title("🎬 YouTube RAG Assistant")
    st.caption(
        "Paste any YouTube URL · Chat with the video · "
        "Every answer cites the exact timestamp"
    )

    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.session_state.error_message = ""

    st.divider()

    if not st.session_state.video_indexed:
        render_url_input()
    else:
        render_chat_interface()


def main() -> None:

    init_session_state()
    st.session_state.pipeline = load_api_client()
    if not st.session_state.pipeline.health():
        st.error(
            "⚠️ Cannot connect to the backend API. "
            "Please try again in a moment."
        )
        st.stop()

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()