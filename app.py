import time
import streamlit as st
from src.config import validate_environment


st.set_page_config(
    page_title = "YouTube RAG Assistant",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)


st.markdown("""
<style>
    /* Tighten chat message padding */
    .stChatMessage { padding: 0.5rem 0; }

    /* Source caption styling */
    .source-label { font-size: 0.8rem; color: #666; }

    /* Remove top margin from first title */
    h1:first-of-type { margin-top: 0; }
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
def load_pipeline():
 
    from src.pipeline import RAGPipeline
    with st.spinner("⏳ Loading AI models — this takes ~10s on first run..."):
        pipeline = RAGPipeline()
    return pipeline



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
    
    pipeline = st.session_state.pipeline

    
    if pipeline.is_video_indexed(url):
        video_id    = pipeline._fetcher.extract_video_id(url)
        chunk_count = pipeline._store.count(video_id)

        st.session_state.video_url         = url
        st.session_state.video_indexed     = True
        st.session_state.video_chunk_count = chunk_count

        st.success(
            f"✅ Already indexed ({chunk_count} chunks) "
            f"— jumping straight to chat!"
        )
        time.sleep(0.8)
        st.rerun()
        return

    
    progress = st.progress(0, text="Starting...")

    try:
        with st.status(
            "🔄 Indexing video...", expanded=True
        ) as status:
            st.write("📄 Fetching transcript from YouTube...")
            progress.progress(15, text="Fetching transcript...")

            st.write("✂️  Chunking into 60-second segments...")
            progress.progress(35, text="Chunking...")

            st.write("🔢 Generating semantic embeddings...")
            progress.progress(60, text="Embedding — this takes ~5s...")

            st.write("💾 Storing in vector database...")
            progress.progress(80, text="Storing...")

    
            st.write("🤖 Running first query to finalise setup...")
            progress.progress(90, text="Finalising...")

            dummy = pipeline.query(
                youtube_url = url,
                question    = "What is this video about?",
            )

            video_id    = dummy.video_id
            chunk_count = pipeline._store.count(video_id)

            st.session_state.video_url         = url
            st.session_state.video_indexed     = True
            st.session_state.video_chunk_count = chunk_count

            progress.progress(100, text="Done!")
            status.update(
                label    = (
                    f"✅ Indexed {chunk_count} chunks — ready to chat!"
                ),
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
                "👋 **Ready!** Ask me anything about this video.\n\n"
                "I'll answer using only its content and show you "
                "the exact timestamp for every claim I make."
            )

    
    for message in st.session_state.messages:
        _render_message(message)

    
    prompt = st.chat_input(
        placeholder = "Ask anything about the video...",
        disabled    = st.session_state.is_loading,
    )

    if prompt and not st.session_state.is_loading:
        _handle_user_question(prompt)


def _render_message(message: dict) -> None:
    
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    
        if message["role"] == "assistant":
            if message.get("answer_grounded") and message.get("sources"):
                _render_sources(message["sources"])
            elif not message.get("answer_grounded", True):
    
                st.caption(
                    "ℹ️ *No relevant sections found in this video "
                    "for this question.*"
                )


def _render_sources(sources: list[dict]) -> None:
    
    if not sources:
        return

    st.caption("📍 **Sources — click to jump to that moment:**")
    cols = st.columns(len(sources))

    for col, source in zip(cols, sources):
        with col:
            st.link_button(
                label               = f"▶ {source['display']}",
                url                 = source["youtube_link"],
                use_container_width = True,
                help                = (
                    f"Open video at {source['start_time']} "
                    f"in a new tab"
                ),
            )


def _handle_user_question(prompt: str) -> None:
    
    from src.pipeline import ConversationTurn

    
    user_message = {
        "role"           : "user",
        "content"        : prompt,
        "sources"        : [],
        "answer_grounded": True,  
    }
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    
    st.session_state.is_loading = True

    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching and generating..."):
            try:
                response = st.session_state.pipeline.query(
                    youtube_url = st.session_state.video_url,
                    question    = prompt,
                    history     = st.session_state.conversation_history,
                )
                error = None
            except Exception as e:
                response = None
                error    = str(e)

    st.session_state.is_loading = False

    
    if error:
        assistant_message = {
            "role"           : "assistant",
            "content"        : (
                f"⚠️ Something went wrong: `{error}`\n\n"
                "Please try again or reload the page."
            ),
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

    
        st.session_state.last_latency      = response.latency
        st.session_state.last_queries_used = response.queries_used

    
    st.session_state.messages.append(assistant_message)
    _render_message(assistant_message)

    
    if not error and response.answer_grounded:
        st.session_state.conversation_history.append(
            ConversationTurn(
                question = prompt,
                answer   = response.answer,
            )
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
    
    try:
        validate_environment()
    except EnvironmentError as e:
        st.error(f"⚠️ Configuration error: {e}")
        st.stop()

    init_session_state()
    st.session_state.pipeline = load_pipeline()

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()