# ============================================================
#  RAG CHATBOT UI - app.py
#  Built with Streamlit
#
#  WHAT'S NEW IN THIS VERSION:
#  ✅ Fixed DOCX support (utf-8 bug fixed)
#  ✅ Accepts PDF, TXT, DOCX files
#  ✅ Proper file extension detection using os.path.splitext
#
#  HOW TO RUN:
#  streamlit run app.py
# ============================================================

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from rag_pipeline import (
    load_embedding_model,
    load_document,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_chunks,
    ask_llm
)

load_dotenv()

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📄",
    layout="centered"
)

# ── STYLING ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  SESSION STATE
#
#  Streamlit reruns entire script on every interaction.
#  session_state persists data between reruns.
#
#  Without it:
#  User uploads PDF → asks question → script reruns → PDF gone!
#
#  With it:
#  chunks, index, model saved → survive every rerun ✅
# ============================================================
# ADD THIS with your other session state initializations:
if "sources" not in st.session_state:
    st.session_state.sources = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "embed_model" not in st.session_state:
    st.session_state.embed_model = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = ""


# ============================================================
#  CACHED MODEL LOADER
#
#  @st.cache_resource loads model ONCE and keeps in memory.
#  Without caching: model reloads on every rerun = 3 sec wait
#  With caching: loads once at startup = instant every time ✅
# ============================================================

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 RAG Chatbot</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a document and ask questions about it</div>',
    unsafe_allow_html=True
)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # ── API KEY CHECK ─────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        st.success("✅ Groq API key loaded")
    else:
        st.error("❌ No GROQ_API_KEY in .env file")
        st.code("GROQ_API_KEY=gsk_your_key_here")
        st.stop()

    st.divider()

    # ── CHUNK SETTINGS ────────────────────────────────────
    st.subheader("📐 Chunk Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Larger = more context. Smaller = more precise retrieval."
    )

    overlap = st.slider(
        "Overlap (words)",
        min_value=0,
        max_value=100,
        value=50,
        step=10,
        help="Words repeated between chunks. Prevents sentences being cut."
    )

    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=6,
        value=3,
        help="How many chunks to send to LLM."
    )

    st.divider()

    # ── FILE UPLOAD ───────────────────────────────────────
    st.subheader("📁 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
          type=["pdf", "txt", "docx", "xlsx", "pptx"],
        # ✅ Now accepts PDF, TXT and DOCX
         help="Upload PDF, TXT, DOCX, XLSX or PPTX"
    )
    st.divider()

    # ── URL INPUT ─────────────────────────────────────────
    st.subheader("🌐 Or Enter Web URLs")
    st.caption("Enter up to 5 URLs (one per line)")

    urls_input = st.text_area(
        "Web URLs",
        placeholder=(
            "https://en.wikipedia.org/wiki/Photosynthesis\n"
            "https://example.com/article"
        ),
        height=120,
        help="Enter up to 5 URLs, one per line"
    )

    if st.button("🌐 Process URLs", use_container_width=True):

        # Split by newlines and clean up
        urls = [
            u.strip()
            for u in urls_input.strip().splitlines()
            if u.strip()
        ]

        # ── VALIDATE URL COUNT ────────────────────────────
        if len(urls) == 0:
            st.error("❌ Please enter at least one URL")

        elif len(urls) > 5:
            st.error(
                f"❌ Maximum 5 URLs allowed. "
                f"You entered {len(urls)}."
            )

        else:
            with st.spinner(f"Scraping {len(urls)} URL(s)..."):
                try:
                    from rag_pipeline import load_url

                    all_text = []
                    sources  = []
                    failed   = []

                    # Process each URL
                    for i, url in enumerate(urls, 1):
                        st.write(f"🌐 Fetching URL {i}/{len(urls)}: {url[:50]}...")
                        try:
                            text, source = load_url(url)
                            all_text.append(
                                f"=== Source: {url} ===\n{text}"
                            )
                            sources.append(url)
                        except Exception as e:
                            failed.append(url)
                            st.warning(f"⚠️ Failed: {url}\n{str(e)}")

                    if not all_text:
                        st.error("❌ All URLs failed to load")
                    else:
                        # Combine all URL content
                        combined_text = "\n\n".join(all_text)

                        # Run RAG pipeline on combined text
                        st.write("✂️ Chunking content...")
                        model  = get_embedding_model()
                        chunks = chunk_text(
                            combined_text,
                            chunk_size=chunk_size,
                            overlap=overlap
                        )

                        st.write("🔢 Creating embeddings...")
                        embeddings = create_embeddings(chunks, model)

                        st.write("🗄️ Building index...")
                        index = build_faiss_index(embeddings)

                        # Save to session state
                        st.session_state.chunks      = chunks
                        st.session_state.faiss_index = index
                        st.session_state.embed_model = model
                        st.session_state.rag_ready   = True
                        st.session_state.doc_name    = (
                            f"{len(sources)} URL(s) scraped"
                        )
                        st.session_state.messages    = []
                        st.session_state.sources     = sources

                        msg = f"✅ Done! {len(chunks)} chunks from {len(sources)} URL(s)"
                        if failed:
                            msg += f" ({len(failed)} failed)"
                        st.success(msg)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # ── PROCESS BUTTON ────────────────────────────────────
    if uploaded_file is not None:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Processing your document..."):
                try:
                    # ── FIX: CORRECT FILE EXTENSION DETECTION ──
                    # OLD (broken):
                    # suffix = ".pdf" if name.endswith(".pdf") else ".txt"
                    # This saved .docx files as .txt → utf-8 crash!
                    #
                    # NEW (correct):
                    # os.path.splitext("resume.docx") → ("resume", ".docx")
                    # We take only the extension part → ".docx"
                    # Temp file gets correct extension → correct loader called
                    _, suffix = os.path.splitext(uploaded_file.name)
                    suffix = suffix.lower()
                    # suffix is now exactly ".pdf", ".txt", or ".docx"

                    # Save to temp file with correct extension
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=suffix
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    # ── RUN RAG PIPELINE ──────────────────
                    st.write("📖 Reading document...")
                    text = load_document(tmp_path)
                    # load_document() sees the .docx extension
                    # and calls load_docx() instead of load_txt()
                    # No more utf-8 crash! ✅

                    st.write("✂️ Chunking text...")
                    chunks = chunk_text(
                        text,
                        chunk_size=chunk_size,
                        overlap=overlap
                    )

                    st.write("📦 Loading embedding model...")
                    model = get_embedding_model()

                    st.write("🔢 Creating embeddings...")
                    embeddings = create_embeddings(chunks, model)

                    st.write("🗄️ Building search index...")
                    index = build_faiss_index(embeddings)

                    # Save to session state
                    st.session_state.chunks      = chunks
                    st.session_state.faiss_index = index
                    st.session_state.embed_model = model
                    st.session_state.rag_ready   = True
                    st.session_state.doc_name    = uploaded_file.name
                    st.session_state.messages    = []

                    # Clean up temp file
                    os.unlink(tmp_path)

                    st.success(
                        f"✅ Ready! {len(chunks)} chunks indexed from "
                        f"{uploaded_file.name}"
                    )

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.divider()

    # ── SAMPLE TEXT BUTTON ────────────────────────────────
    if st.button("📝 Use Sample Text", use_container_width=True):
        with st.spinner("Loading sample text..."):
            sample_text = """
            Photosynthesis is the process by which plants use sunlight,
            water, and carbon dioxide to produce oxygen and energy in the
            form of sugar. This process takes place in the chloroplasts,
            specifically using the green pigment called chlorophyll.
            Photosynthesis is crucial for life on Earth as it forms the
            base of most food chains and produces the oxygen we breathe.

            The water cycle describes how water evaporates from the surface
            of the earth, rises into the atmosphere, cools and condenses
            into rain or snow in clouds, and falls again to the surface as
            precipitation. Water on earth is constantly moving through
            evaporation, condensation, and precipitation.

            Gravity is a fundamental force of nature that attracts objects
            with mass toward each other. On Earth, gravity gives weight to
            physical objects and causes them to fall toward the ground when
            dropped. Isaac Newton first described gravity mathematically,
            while Albert Einstein later explained it as the curvature of
            spacetime in his theory of general relativity.

            The human digestive system breaks down food into nutrients that
            the body can absorb and use. It begins in the mouth where food
            is chewed and mixed with saliva. The food then travels down the
            esophagus to the stomach, where it is mixed with gastric acids.
            The small intestine absorbs most nutrients, while the large
            intestine absorbs water and forms waste.
            """

            model      = get_embedding_model()
            chunks     = chunk_text(
                sample_text,
                chunk_size=chunk_size,
                overlap=overlap
            )
            embeddings = create_embeddings(chunks, model)
            index      = build_faiss_index(embeddings)

            st.session_state.chunks      = chunks
            st.session_state.faiss_index = index
            st.session_state.embed_model = model
            st.session_state.rag_ready   = True
            st.session_state.doc_name    = "Sample Text"
            st.session_state.messages    = []

            st.success(f"✅ Sample loaded! {len(chunks)} chunks ready.")

    # ── SHOW CURRENT DOC ──────────────────────────────────
    if st.session_state.rag_ready:
        st.divider()
        st.caption("📄 Current document:")
        st.info(st.session_state.doc_name)
        st.caption(
            f"📦 {len(st.session_state.chunks)} chunks indexed"
        )


# ── MAIN CHAT AREA ───────────────────────────────────────────

if not st.session_state.rag_ready:
    # Instructions when no document loaded
    st.info("👈 Upload a document or click 'Use Sample Text' to start!")

    with st.expander("ℹ️ How does this work?"):
        st.markdown("""
        **Phase 1 — Indexing (when you upload):**
        1. Document is read and split into chunks
        2. Each chunk → vector (384 numbers)
        3. Vectors stored in FAISS

        **Phase 2 — Querying (when you ask):**
        1. Question → vector
        2. FAISS finds 3 most similar chunks
        3. Chunks + question → Groq (Llama 3)
        4. Answer based ONLY on your document

        **Supported files:** PDF, TXT, DOCX
        """)

else:
    # ── DISPLAY CHAT HISTORY ─────────────────────────────
     for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant":

                # Show source links if they exist
                if message.get("sources"):
                    st.markdown("**📎 Sources:**")
                    for source in message["sources"]:
                        st.markdown(f"- [{source}]({source})")

                # Show retrieved chunks
                if "chunks" in message:
                    with st.expander("📄 View retrieved chunks"):
                        for i, chunk in enumerate(
                            message["chunks"], 1
                        ):
                            st.caption(f"Chunk {i}:")
                            st.text(chunk[:300] + "...")

    # ── CHAT INPUT ────────────────────────────────────────
     if query := st.chat_input(
        "Ask a question about your document..."
    ):
        # Show user message
        with st.chat_message("user"):
            st.write(query)

        # Save to history
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Step 5: Retrieve relevant chunks
                relevant_chunks = retrieve_chunks(
                    query,
                    st.session_state.embed_model,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    top_k=top_k
                )

                # Step 6: Generate answer
                answer = ask_llm(
                    query,
                    relevant_chunks,
                    api_key
                )

            # Display answer
            st.write(answer)

            # ── SHOW SOURCE LINKS ─────────────────────────
            # If content came from URLs, show which ones
            if st.session_state.sources:
                st.markdown("**📎 Sources:**")
                for source in st.session_state.sources:
                    st.markdown(f"- [{source}]({source})")

            # Show retrieved chunks
            with st.expander("📄 View retrieved chunks"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.caption(f"Chunk {i} (used as context):")
                    st.text(chunk[:300] + "...")

        # Save to history
        st.session_state.messages.append({
            "role"  : "assistant",
            "content": answer,
            "chunks": relevant_chunks,
            "sources": st.session_state.sources.copy()
        })
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": relevant_chunks
        })