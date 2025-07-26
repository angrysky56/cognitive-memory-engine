#!/usr/bin/env python3
"""
Debug initialization to trace missing component connections
"""
import asyncio
import logging
import sys
import traceback
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, 'src')

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
)

async def debug_initialization():
    try:
        print("=== STEP 1: Import config ===")
        from cognitive_memory_engine.config import (
            get_cloud_provider_config,
            load_env_config,
        )
        print("✓ Config imports successful")

        print("=== STEP 2: Load config ===")
        config = load_env_config()
        print(f"✓ Config loaded: {config.llm_model}")

        cloud_config = get_cloud_provider_config()
        print(f"✓ Cloud config loaded: {type(cloud_config)}")

        print("=== STEP 3: Import engine ===")
        from cognitive_memory_engine.core.engine import CognitiveMemoryEngine
        print("✓ Engine import successful")

        print("=== STEP 4: Create engine ===")
        engine = CognitiveMemoryEngine(config)
        print("✓ Engine created")

        print("Initial component states:")
        print(f"  narrative_builder: {engine.narrative_builder}")
        print(f"  document_builder: {engine.document_builder}")
        print(f"  vector_manager: {engine.vector_manager}")
        print(f"  initialized: {engine.initialized}")

        print("=== STEP 5: Initialize engine with detailed tracing ===")

        # Manually trace through initialization steps
        if engine.initialized:
            print("Engine already initialized, skipping")
            return

        print("Setting up data directory...")
        data_dir = Path(engine.config.data_directory)
        data_dir.mkdir(exist_ok=True)
        print(f"✓ Data directory: {data_dir}")

        print("Testing component imports...")
        from cognitive_memory_engine.comprehension.document_knowledge_builder import (
            DocumentKnowledgeBuilder,
        )
        from cognitive_memory_engine.comprehension.narrative_tree_builder import (
            NarrativeTreeBuilder,
        )
        from cognitive_memory_engine.comprehension.temporal_organizer import (
            TemporalOrganizer,
        )
        from cognitive_memory_engine.production.response_generator import (
            ResponseGenerator,
        )
        from cognitive_memory_engine.storage.document_store import DocumentStore
        from cognitive_memory_engine.storage.rtm_graphs import RTMGraphStore
        from cognitive_memory_engine.storage.temporal_library import TemporalLibrary
        from cognitive_memory_engine.storage.vector_store import VectorStore
        from cognitive_memory_engine.workspace.context_assembler import ContextAssembler
        from cognitive_memory_engine.workspace.svg_vector_manager import (
            SVGVectorManager,
        )
        from cognitive_memory_engine.workspace.vector_manager import VectorManager
        print("✓ All component imports successful")

        print("Testing component creation step by step...")

        print("1. Creating VectorStore...")
        try:
            vector_store = VectorStore({
                "persist_directory": str(data_dir / "chroma_db"),
                "collection_name": "cme_memory"
            })
            await vector_store.initialize()
            print("✓ VectorStore created and initialized")
        except Exception as e:
            print(f"✗ VectorStore failed: {e}")
            traceback.print_exc()

        print("2. Creating TemporalLibrary...")
        try:
            temporal_library = TemporalLibrary(str(data_dir / "temporal"))
            await temporal_library.initialize()
            print("✓ TemporalLibrary created and initialized")
        except Exception as e:
            print(f"✗ TemporalLibrary failed: {e}")
            traceback.print_exc()

        print("3. Creating RTMGraphStore...")
        try:
            rtm_store = RTMGraphStore(str(data_dir / "rtm_graphs"))
            print("✓ RTMGraphStore created")
        except Exception as e:
            print(f"✗ RTMGraphStore failed: {e}")
            traceback.print_exc()

        print("4. Creating DocumentStore...")
        try:
            document_store = DocumentStore(str(data_dir / "document_knowledge"))
            print("✓ DocumentStore created")
        except Exception as e:
            print(f"✗ DocumentStore failed: {e}")
            traceback.print_exc()

        print("5. Creating DocumentKnowledgeBuilder...")
        try:
            document_builder = DocumentKnowledgeBuilder(
                cloud_config=cloud_config.__dict__,
                document_store=document_store
            )
            print("✓ DocumentKnowledgeBuilder created")
        except Exception as e:
            print(f"✗ DocumentKnowledgeBuilder failed: {e}")
            traceback.print_exc()

        print("6. Creating NarrativeTreeBuilder...")
        try:
            narrative_builder = NarrativeTreeBuilder(
                cloud_config=cloud_config.__dict__,
                rtm_store=rtm_store,
                config=engine.config.rtm_config
            )
            print("✓ NarrativeTreeBuilder created")
        except Exception as e:
            print(f"✗ NarrativeTreeBuilder failed: {e}")
            traceback.print_exc()

        print("7. Creating VectorManager...")
        try:
            if engine.config.vector_manager == 'svg':
                vector_manager = SVGVectorManager(
                    storage_path=str(data_dir / "vectors"),
                    embedding_model=engine.config.embedding_model,
                    config=engine.config.neural_gain_config,
                    svg_config=engine.config.svg_config
                )
            else:
                vector_manager = VectorManager(
                    storage_path=str(data_dir / "vectors"),
                    embedding_model=engine.config.embedding_model,
                    config=engine.config.neural_gain_config
                )
            print(f"✓ VectorManager created ({type(vector_manager).__name__})")
        except Exception as e:
            print(f"✗ VectorManager failed: {e}")
            traceback.print_exc()

        print("8. Creating TemporalOrganizer...")
        try:
            temporal_organizer = TemporalOrganizer(
                temporal_library=temporal_library,
                vector_manager=vector_manager,
                rtm_store=rtm_store,
                config=engine.config.neural_gain_config
            )
            print("✓ TemporalOrganizer created")
        except Exception as e:
            print(f"✗ TemporalOrganizer failed: {e}")
            traceback.print_exc()

        print("9. Creating ContextAssembler...")
        try:
            context_assembler = ContextAssembler(
                vector_manager=vector_manager,
                rtm_store=rtm_store,
                temporal_library=temporal_library,
                max_context_length=engine.config.max_context_length
            )
            print("✓ ContextAssembler created")
        except Exception as e:
            print(f"✗ ContextAssembler failed: {e}")
            traceback.print_exc()

        print("10. Creating ResponseGenerator...")
        try:
            response_generator = ResponseGenerator(
                cloud_config=cloud_config.__dict__,
                max_response_length=1000,
                temperature=0.7
            )
            print("✓ ResponseGenerator created")
        except Exception as e:
            print(f"✗ ResponseGenerator failed: {e}")
            traceback.print_exc()

        print("=== All manual component creation complete ===")

        print("Now testing actual engine initialization...")
        await engine.initialize()

        print("After engine.initialize() component states:")
        print(f"  narrative_builder: {engine.narrative_builder}")
        print(f"  document_builder: {engine.document_builder}")
        print(f"  vector_manager: {engine.vector_manager}")
        print(f"  temporal_organizer: {engine.temporal_organizer}")
        print(f"  context_assembler: {engine.context_assembler}")
        print(f"  response_generator: {engine.response_generator}")
        print(f"  initialized: {engine.initialized}")

        if engine.narrative_builder is None:
            print("❌ CRITICAL: narrative_builder is still None after initialization!")
        if engine.document_builder is None:
            print("❌ CRITICAL: document_builder is still None after initialization!")
        if engine.vector_manager is None:
            print("❌ CRITICAL: vector_manager is still None after initialization!")

        print("=== Debug complete ===")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_initialization())
