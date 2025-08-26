import streamlit as st
import tempfile
import time
import os
from typing import Optional
from unstructured.partition.auto import partition
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# -------- Configuration --------
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MAX_PREVIEW_CHARS = 2000

# -------- Helper Functions --------
@st.cache_data
def extract_text(file) -> str:
    """Extract text from uploaded file with caching for better performance."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        elements = partition(filename=tmp_path, languages=["eng"])
        text = "\n".join([el.text for el in elements if el.text])
        
        # Clean up temp file
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

def load_job_description(file_path: str) -> Optional[str]:
    """Load job description from file with error handling."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            st.warning(f"Job description file '{file_path}' not found. Please upload one manually.")
            return None
    except Exception as e:
        st.error(f"Error loading job description: {str(e)}")
        return None

def create_vectorstore(text: str, embedding_model: str = EMBEDDING_MODEL) -> Optional[FAISS]:
    """Create vectorstore from text with error handling."""
    try:
        if not text.strip():
            st.error("No text content found to create embeddings.")
            return None
            
        splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.create_documents([text])
        
        if not chunks:
            st.error("No chunks created from the text.")
            return None
            
        embeddings = OllamaEmbeddings(model=embedding_model)
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def analyze_resume(job_description: str, resume: str) -> str:
    """Analyze resume against job description using LLM."""
    try:
        llm = OllamaLLM(model=LLM_MODEL, temperature=0)
        
        prompt = PromptTemplate(
            template="""
            You are an expert resume parser and hiring assistant. Analyze the following job description and resume to determine if the candidate should be shortlisted.

            Please provide a structured analysis including:
            1. **RECOMMENDATION**: Should the candidate be SHORTLISTED or NOT SHORTLISTED?
            2. **MATCH SCORE**: Rate the match from 1-10 (10 being perfect match)
            3. **KEY STRENGTHS**: What makes this candidate suitable for the role?
            4. **GAPS IDENTIFIED**: What requirements are missing or could be stronger?
            5. **SPECIFIC EXAMPLES**: Cite specific skills/experiences from the resume that align with job requirements
            6. **SUMMARY**: Brief conclusion with reasoning

            JOB DESCRIPTION:
            {job_description}

            RESUME:
            {resume}

            Provide a detailed, objective analysis:
            """,
            input_variables=['job_description', 'resume']
        )
        
        final_prompt = prompt.format(job_description=job_description, resume=resume)
        return llm.invoke(final_prompt)
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# -------- Initialize Session State --------
def initialize_session_state():
    """Initialize all session state variables."""
    if "job_vectorstore" not in st.session_state:
        st.session_state.job_vectorstore = None
    if "resume_vectorstore" not in st.session_state:
        st.session_state.resume_vectorstore = None
    if "job_description_text" not in st.session_state:
        st.session_state.job_description_text = ""
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

# -------- Main Application --------
def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Resume Parser & Matcher", 
        page_icon="📄", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Header
    st.title("📄 AI Resume Parser & Matcher")
    st.markdown("Upload a resume and compare it against job descriptions using AI-powered analysis.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        embedding_model = st.selectbox(
            "Embedding Model",
            [EMBEDDING_MODEL, "all-minilm", "mxbai-embed-large"],
            help="Choose the embedding model for text processing"
        )
        
        llm_model = st.selectbox(
            "Language Model",
            [LLM_MODEL, "llama2", "mistral", "codellama"],
            help="Choose the LLM for resume analysis"
        )
        
        st.header("📊 Analysis History")
        if st.session_state.analysis_history:
            for i, analysis in enumerate(st.session_state.analysis_history[-3:], 1):
                st.text_area(f"Analysis {i}", analysis[:200] + "...", height=100, disabled=True)
        
        if st.button("Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Job Description")
        
        # Option to load from file or paste directly
        job_input_method = st.radio(
            "Choose input method:",
            ["Load from file.txt", "Upload file", "Paste directly"],
            horizontal=True
        )
        
        if job_input_method == "Load from file.txt":
            job_description = load_job_description('file.txt')
            if job_description:
                st.success("✅ Job description loaded from file.txt")
                st.text_area("Job Description Preview", job_description[:500] + "...", height=150, disabled=True)
                st.session_state.job_description_text = job_description
            
        elif job_input_method == "Upload file":
            job_file = st.file_uploader("Upload job description", type=["txt", "pdf", "docx"], key="job_file")
            if job_file:
                with st.spinner("Extracting job description..."):
                    job_description = extract_text(job_file)
                    if job_description:
                        st.success("✅ Job description extracted")
                        st.text_area("Job Description Preview", job_description[:500] + "...", height=150, disabled=True)
                        st.session_state.job_description_text = job_description
        
        elif job_input_method == "Paste directly":
            job_description = st.text_area(
                "Paste job description here:",
                height=200,
                placeholder="Paste the job description text here..."
            )
            if job_description:
                st.session_state.job_description_text = job_description
        
        # Create job description vectorstore
        if st.session_state.job_description_text and not st.session_state.job_vectorstore:
            with st.spinner("Processing job description..."):
                st.session_state.job_vectorstore = create_vectorstore(
                    st.session_state.job_description_text, 
                    embedding_model
                )
    
    with col2:
        st.header("📄 Resume")
        
        uploaded_resume = st.file_uploader(
            "Upload resume", 
            type=["pdf", "docx", "doc", "txt"],
            help="Supported formats: PDF, Word documents, and text files"
        )
        
        if uploaded_resume:
            # Reset vectorstore when new file is uploaded
            if st.session_state.resume_vectorstore is None:
                with st.spinner("Extracting resume text..."):
                    resume_text = extract_text(uploaded_resume)
                    
                if resume_text:
                    st.success("✅ Resume text extracted")
                    st.text_area(
                        "Resume Preview", 
                        resume_text[:MAX_PREVIEW_CHARS] + ("..." if len(resume_text) > MAX_PREVIEW_CHARS else ""),
                        height=200,
                        disabled=True
                    )
                    st.session_state.resume_text = resume_text
                    
                    # Create resume vectorstore
                    with st.spinner("Processing resume..."):
                        st.session_state.resume_vectorstore = create_vectorstore(resume_text, embedding_model)
    
    # Analysis section
    st.header("🤖 AI Analysis")
    
    if st.session_state.job_description_text and st.session_state.resume_text:
        col_analyze, col_reset = st.columns([3, 1])
        
        with col_analyze:
            if st.button("🚀 Analyze Resume Match", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the resume against job requirements..."):
                    time.sleep(1)  # Small delay for UX
                    
                    # Get context from vectorstores if available
                    if st.session_state.job_vectorstore and st.session_state.resume_vectorstore:
                        job_retriever = st.session_state.job_vectorstore.as_retriever(
                            search_type='similarity', 
                            search_kwargs={"k": 4}
                        )
                        resume_retriever = st.session_state.resume_vectorstore.as_retriever(
                            search_type='similarity', 
                            search_kwargs={"k": 4}
                        )
                        
                        job_context = '\n\n'.join([doc.page_content for doc in job_retriever.get_relevant_documents("")])
                        resume_context = '\n\n'.join([doc.page_content for doc in resume_retriever.get_relevant_documents("")])
                    else:
                        job_context = st.session_state.job_description_text
                        resume_context = st.session_state.resume_text
                    
                    # Perform analysis
                    result = analyze_resume(job_context, resume_context)
                    
                    # Store in history
                    st.session_state.analysis_history.append(result)
                
                # Display results
                st.subheader("🎯 Analysis Results")
                st.markdown(result)
                
                # Option to download results
                st.download_button(
                    label="📥 Download Analysis",
                    data=result,
                    file_name=f"resume_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col_reset:
            if st.button("🔄 Reset All", help="Clear all uploaded data"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    else:
        st.info("👆 Please upload both a job description and a resume to begin analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip**: This tool uses local LLMs via Ollama. Ensure you have the required models installed: "
        f"`{embedding_model}` and `{llm_model}`"
    )

if __name__ == "__main__":
    main()