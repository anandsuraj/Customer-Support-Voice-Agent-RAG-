# Standard library imports
import os
import tempfile
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Third-party library imports
import streamlit as st
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from firecrawl import FirecrawlApp, AsyncFirecrawlApp, ScrapeOptions
from fastembed import TextEmbedding
from openai import AsyncOpenAI

# Local application imports
from agents import Agent, Runner

import asyncio


load_dotenv()


def init_session_state():
    defaults = {
        "initialized": False,
        "qdrant_url": "",
        "qdrant_api_key": "",
        "firecrawl_api_key": "",
        "openai_api_key": "",
        "doc_url": "",
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "selected_voice": "coral",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


async def sidebar_config():
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL", value=st.session_state.qdrant_url, type="password"
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key", value=st.session_state.qdrant_api_key, type="password"
        )
        st.session_state.firecrawl_api_key = st.text_input(
            "Firecrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password",
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key", value=st.session_state.openai_api_key, type="password"
        )

        st.markdown("---")
        st.session_state.doc_url = st.text_input(
            "Documentation URL",
            value=st.session_state.doc_url,
            placeholder="https://docs.example.com",
        )

        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")
        voices = [
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "onyx",
            "nova",
            "sage",
            "shimmer",
            "verse",
        ]
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
            help="Choose the voice for the audio response",
        )

        if st.button("Initialize System", type="primary"):
            if all(
                [
                    st.session_state.qdrant_url,
                    st.session_state.qdrant_api_key,
                    st.session_state.firecrawl_api_key,
                    st.session_state.openai_api_key,
                    st.session_state.doc_url,
                ]
            ):
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    try:
                        st.markdown("🔄 Setting up Qdrant connection...")
                        client, embedding_model = setup_qdrant_collection(
                            st.session_state.qdrant_url, st.session_state.qdrant_api_key
                        )
                        st.session_state.client = client
                        st.session_state.embedding_model = embedding_model
                        st.markdown("✅ Qdrant setup complete!")

                        st.markdown("🔄 Crawling documentation pages...")
                        pages = await crawl_documentation(
                            st.session_state.firecrawl_api_key, st.session_state.doc_url
                        )
                        st.markdown(
                            f"✅ Crawled {len(pages)} documentation pages!")

                        store_embeddings(
                            client, embedding_model, pages, "docs_embeddings"
                        )

                        processor_agent = setup_agents(
                            st.session_state.openai_api_key
                        )
                        st.session_state.processor_agent = processor_agent

                        st.session_state.setup_complete = True
                        st.success("✅ System initialized successfully!")

                    except Exception as e:
                        st.error(f"Error during setup: {str(e)}")
            else:
                st.error("Please fill in all the required fields!")


def setup_qdrant_collection(
    qdrant_url: str, qdrant_api_key: str, collection_name: str = "docs_embeddings"
):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim, distance=Distance.COSINE),
        )
    except Exception as e:
        if "already exists" not in str(e):
            raise e

    return client, embedding_model


async def crawl_documentation(firecrawl_api_key: str, url: str, output_dir: Optional[str] = None) -> List[Dict]:
    app = AsyncFirecrawlApp(api_key=firecrawl_api_key)
    pages = []

    try:
        response = await app.crawl_url(
            url=url,
            limit=5,
            ignore_sitemap=True,
            allow_backward_links=True,
            scrape_options=ScrapeOptions(
                formats=["markdown"], onlyMainContent=True),
        )

        print("Firecrawl response:", response)

        if not response or not getattr(response, "success", False):
            raise Exception("Firecrawl API call was not successful.")

        data = getattr(response, "data", None)
        if not data:
            raise Exception("No data returned from Firecrawl API")

        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            elif not os.path.isdir(output_dir):
                raise ValueError(
                    f"Output directory {output_dir} is not a directory")

        for page in data:
            content = getattr(page, "markdown", "")
            metadata = getattr(page, "metadata", {})
            source_url = metadata.get("sourceURL", "")

            if output_dir and content:
                filename = f"{uuid.uuid4()}.md"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            pages.append(
                {
                    "content": content,
                    "url": source_url,
                    "metadata": {
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "language": metadata.get("language", "en"),
                        "crawl_date": datetime.now().isoformat(),
                    },
                }
            )

    except Exception as e:
        print(f"Error during crawling: {e}")

    return pages

def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    pages: List[Dict],
    collection_name: str,
):
    for page in pages:
        embedding = list(embedding_model.embed([page["content"]]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": page["content"],
                        "url": page["url"],
                        **page["metadata"],
                    },
                )
            ],
        )


def setup_agents(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key

    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""You are a helpful documentation assistant. Your task is to:
        1. Analyze the provided documentation content
        2. Answer the user's question clearly and concisely
        3. Include relevant examples when available
        4. Cite the source URLs when referencing specific content
        5. Keep responses natural and conversational
        6. Format your response in a way that's easy to speak out loud""",
        model="gpt-4o",
    )

    return processor_agent

async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    processor_agent: Agent,
    collection_name: str,
    openai_api_key: str,
):
    try:
        query_embedding = list(embedding_model.embed([query]))[0]
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=3,
            with_payload=True,
        )

        search_results = (
            search_response.points if hasattr(
                search_response, "points") else []
        )

        if not search_results:
            raise Exception(
                "No relevant documents found in the vector database")

        context = "Based on the following documentation:\n\n"
        for result in search_results:
            payload = result.payload
            if not payload:
                continue
            url = payload.get("url", "Unknown URL")
            content = payload.get("content", "")
            context += f"From {url}:\n{content}\n\n"

        context += f"\nUser Question: {query}\n\n"
        context += (
            "Please provide a clear, concise answer that can be easily spoken out loud."
        )

        processor_result = await Runner.run(processor_agent, context)
        processor_response = processor_result.final_output

        # Generate audio using OpenAI TTS API directly
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        audio_response = await async_openai.audio.speech.create(
            model="tts-1",
            voice=st.session_state.selected_voice,
            input=processor_response,
            response_format="mp3",
        )

        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"response_{uuid.uuid4()}.mp3")

        with open(audio_path, "wb") as f:
            f.write(audio_response.content)

        return {
            "status": "success",
            "text_response": processor_response,
            "audio_path": audio_path,
            "sources": [
                r.payload.get("url", "Unknown URL") for r in search_results if r.payload
            ],
            "query_details": {
                "vector_size": len(query_embedding),
                "results_found": len(search_results),
                "collection_name": collection_name,
            },
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "query": query}


def run_streamlit():
    st.set_page_config(
        page_title="Customer Support Voice Agent", page_icon="🎙️", layout="wide"
    )

    init_session_state()
    asyncio.run(sidebar_config())

    st.title("🎙️ Customer Support Voice Agent")
    st.markdown(
        """
        **Get quick voice and text answers from any document url**

        Just follow these steps:
        1. Set your API key in the sidebar  
        2. Enter the link to the documentation or any website you want to explore  
        3. Ask a question below and get helpful responses instantly
        """
    )

    query = st.text_input(
        "What would you like to know about the documentation?",
        placeholder="e.g., How do I authenticate API requests?",
        disabled=not st.session_state.setup_complete,
    )

    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                st.markdown(
                    "🔄 Searching documentation and generating response...")
                result = asyncio.run(
                    process_query(
                        query,
                        st.session_state.client,
                        st.session_state.embedding_model,
                        st.session_state.processor_agent,
                        "docs_embeddings",
                        st.session_state.openai_api_key,
                    )
                )

                if result["status"] == "success":
                    status.update(label="✅ Query processed!", state="complete")

                    st.markdown("### Response:")
                    st.write(result["text_response"])

                    if "audio_path" in result:
                        st.markdown(
                            f"### 🔊 Audio Response (Voice: {st.session_state.selected_voice})"
                        )
                        st.audio(result["audio_path"],
                                 format="audio/mp3", start_time=0)

                        with open(result["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.download_button(
                                label="📥 Download Audio Response",
                                data=audio_bytes,
                                file_name=f"voice_response_{st.session_state.selected_voice}.mp3",
                                mime="audio/mp3",
                            )

                    st.markdown("### Sources:")
                    for source in result["sources"]:
                        st.markdown(f"- {source}")
                else:
                    status.update(
                        label="❌ Error processing query", state="error")
                    st.error(
                        f"Error: {result.get('error', 'Unknown error occurred')}")

            except Exception as e:
                status.update(label="❌ Error processing query", state="error")
                st.error(f"Error processing query: {str(e)}")

    elif not st.session_state.setup_complete:
        st.info("👈 Please configure the system using the sidebar first!")


if __name__ == "__main__":
    run_streamlit()