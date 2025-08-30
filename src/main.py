import logging
from rag import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    logging.info("Starting RAG pipeline")
    rag = RAGPipeline()
    rag.setup_pipeline()

if __name__ == "__main__":
    main()