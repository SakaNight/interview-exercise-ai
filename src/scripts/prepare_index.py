import os
import sys
import argparse
import logging
from pathlib import Path
from rag import RAGPipeline
from settings import settings

# Allow script to be executed from outside src/
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("prepare_index")

# Change relative path to absolute path
def resolve_path(p: str | None, default_from_settings: str) -> str:
    path = p or default_from_settings
    if not os.path.isabs(path):
        # Relative path - relative to project root
        path = os.path.join(PROJECT_ROOT.as_posix(), path)
    return path

def main():
    parser = argparse.ArgumentParser(
        description="Build/refresh FAISS index from docs.json for the RAG pipeline."
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=None,
        help="Path to docs.json (defaults to settings.docs_path)"
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to FAISS index file (defaults to settings.index_path)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index even if an index file already exists"
    )
    args = parser.parse_args()

    # Resolve paths (relative -> absolute)
    docs_path = resolve_path(args.docs, settings.docs_path)
    index_path = resolve_path(args.index, settings.index_path)

    # Set RAG path/behavior
    logger.info("Docs path  : %s", docs_path)
    logger.info("Index path : %s", index_path)
    logger.info("Rebuild    : %s", args.rebuild or settings.rebuild_index)

    # Create RAG instance
    rag = RAGPipeline()
    rag.index_path = index_path

    rag.load_documents(docs_path)

    # Prepare texts
    texts = rag.prepare_documents()
    if not texts:
        raise RuntimeError("No documents to index. Check docs.json content.")

    # Decide whether to rebuild
    need_rebuild = args.rebuild or settings.rebuild_index or (not os.path.exists(index_path))
    if need_rebuild:
        logger.info("Building embeddings & FAISS index ...")
        # Create embeddings & FAISS index
        embeddings = rag.create_embeddings(texts)
        rag.build_index(embeddings)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        rag.save_index(index_path)
        logger.info("Index built and saved to: %s", index_path)
    else:
        logger.info("Index exists. Loading existing index from: %s", index_path)
        # Load existing index
        if not rag.load_index(index_path):
            logger.warning("Existing index not found or failed to load. Rebuilding...")
            # Create embeddings & FAISS index
            embeddings = rag.create_embeddings(texts)
            rag.build_index(embeddings)
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            rag.save_index(index_path)
            logger.info("Index rebuilt and saved to: %s", index_path)

    logger.info("Done. Total docs: %d", len(rag.documents))

if __name__ == "__main__":
    main()