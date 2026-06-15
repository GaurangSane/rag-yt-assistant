import streamlit as st
from src.config import validate_environment

st.set_page_config(
    page_title="Youtube RAG assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state() -> None:
    defaults = {
        "pipeline"         : None,                          
        "video_url"        : "",      
        "video_indexed"    : False,   
        "video_chunk_count": 0,       
        "messages"         : [],      
        "conversation_history": [],   
        "is_loading"       : False,   
        "error_message"    : "",      
        "last_query_ms"    : 0.0,     
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

@st.cache_resource
def load_pipeline():
    from src.pipeline import RAGPipeline

    with st.spinner("⏳ Loading AI models (first load only)..."):
        pipeline = RAGPipeline()

    return pipeline            

def render_sidebar() -> None:
    """
    Render the sidebar with app info and debug stats.

    The sidebar is persistent across all page states —
    good place for metadata that doesn't change with chat.
    """
    with st.sidebar:
        st.title("🎬 YT RAG Assistant")
        st.caption("Ask anything about any YouTube video")

        st.divider()

        with st.expander("⚙️ How it works", expanded=False):
            st.markdown("""
**9-step RAG pipeline:**

1. 📄 Fetch YouTube transcript
2. ✂️  Split into 60s chunks
3. 🔢 Embed with all-mpnet-base-v2
4. 💾 Store in ChromaDB
5. 🔄 Rewrite your question (×3)
6. 🔍 Hybrid search (semantic + BM25)
7. 🎯 Rerank with cross-encoder
8. 📝 Build structured prompt
9. 🤖 Generate with LLaMA3 via Groq
            """)

        st.divider()


        st.subheader("📊 Session Info")

        if st.session_state.video_indexed:
            st.success(f"✅ Video loaded")
            st.metric(
                "Chunks indexed",
                st.session_state.video_chunk_count
            )
            st.metric(
                "Messages",
                len(st.session_state.messages)
            )
            if st.session_state.last_query_ms:
                st.metric(
                    "Last query",
                    f"{st.session_state.last_query_ms:.0f}ms"
                )
        else:
            st.info("No video loaded yet")

        st.divider()


        if st.session_state.messages:
            if st.button(
                "🗑️ Clear conversation",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.messages            = []
                st.session_state.conversation_history = []
                st.session_state.last_query_ms        = 0.0
                st.rerun()   

        if st.session_state.video_indexed:
            if st.button(
                "🔄 Load different video",
                use_container_width=True,
                type="secondary",
            ):
                st.session_state.video_url         = ""
                st.session_state.video_indexed     = False
                st.session_state.video_chunk_count = 0
                st.session_state.messages          = []
                st.session_state.conversation_history = []
                st.session_state.error_message     = ""
                st.rerun()

def render_main()->None:

    st.title("🎬 YouTube RAG Assistant")
    st.caption(
        "Paste any YouTube URL and chat with the video content. "
        "Every answer cites the exact timestamp."
    )

    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        st.session_state.error_message = ""

    st.divider()

    if not st.session_state.video_indexed:
        render_url_input()
    else:
        render_chat_interface()

def render_url_input() -> None:
    st.subheader("📎 Load a YouTube Video")

    col1, col2 = st.columns([4, 1])

    with col1:
        url = st.text_input(
            label       = "YouTube URL",
            placeholder = "https://www.youtube.com/watch?v=...",
            label_visibility = "collapsed",
        )

    with col2:
        load_clicked = st.button(
            "Load Video",
            type             = "primary",
            use_container_width = True,
        )

    if load_clicked and url:
        if not url and url.strip():
            st.warning("⚠️ Please enter a YouTube URL first.")

        else:
            _run_ingestion(url.strip())    

    st.divider()
    st.caption("🧪 Try these example videos:")

    examples = [
        ("3Blue1Brown — Neural Networks",
         "https://www.youtube.com/watch?v=aircAruvnKk"),
        ("Andrej Karpathy — Intro to LLMs",
         "https://www.youtube.com/watch?v=zjkBMFhNj_g"),
    ]

    for title, example_url in examples:
        st.code(example_url, language=None)
        st.caption(f"↑ {title}")   

def _run_ingestion(url:str)->None:
    pipeline = st.session_state.pipeline
    if pipeline.is_video_indexed(url):
        video_id    = pipeline._fetcher.extract_video_id(url)
        chunk_count = pipeline._store.count(video_id)

        st.session_state.video_url         = url
        st.session_state.video_indexed     = True
        st.session_state.video_chunk_count = chunk_count

        st.success(
            f"✅ Video already indexed! "
            f"({chunk_count} chunks) — jumping straight to chat."
        )
        import time; time.sleep(0.8)
        st.rerun()
        return

    progress_bar = st.progress(0, text="Starting ingestion...")

    try:
        with st.status("🔄 Indexing video...", expanded=True) as status:

    
            st.write("📄 Fetching transcript from YouTube...")
            progress_bar.progress(10, text="Fetching transcript...")

            st.write("✂️  Chunking transcript...")
            progress_bar.progress(30, text="Chunking...")

            st.write("🔢 Generating embeddings...")
            progress_bar.progress(55, text="Embedding chunks...")

            st.write("💾 Storing in vector database...")
            progress_bar.progress(80, text="Storing...")

            dummy_response = pipeline.query(
                youtube_url = url,
                question    = "What is this video about?",
            )

            progress_bar.progress(100, text="Complete!")

            st.session_state.video_url         = url
            st.session_state.video_indexed     = True
            st.session_state.video_chunk_count = len(
                dummy_response.sources
            ) if dummy_response.sources else dummy_response.citation_count

            video_id    = dummy_response.video_id
            chunk_count = pipeline._store.count(video_id)
            st.session_state.video_chunk_count = chunk_count

            status.update(
                label    = f"✅ Video indexed! {chunk_count} chunks created.",
                state    = "complete",
                expanded = False,
            )

        
        import time; time.sleep(1.0)
        st.rerun()   

    except Exception as e:
        progress_bar.empty()   

    
        error_str = str(e)

        if "TranscriptNotAvailable" in error_str or "disabled" in error_str:
            st.error(
                "❌ This video doesn't have captions/subtitles available. "
                "Try a different video — educational content usually works best."
            )
        elif "InvalidYouTubeURL" in error_str or "video ID" in error_str:
            st.error(
                "❌ Couldn't parse that URL. "
                "Make sure it's a full YouTube URL like: "
                "https://www.youtube.com/watch?v=VIDEO_ID"
            )
        elif "GROQ_API_KEY" in error_str:
            st.error(
                "❌ Groq API key not found. "
                "Check your .env file has GROQ_API_KEY set."
            )
        else:
            st.error(f"❌ Ingestion failed: {error_str}")

def render_chat_interface() -> None:
    st.subheader("💬 Chat with the Video")

    st.info(
        "🚧 Chat interface coming in Stage 3. "
        "Video loading coming in Stage 2."
    )

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