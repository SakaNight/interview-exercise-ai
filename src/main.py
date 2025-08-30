import logging
from rag import RAGPipeline
from exceptions import RAGError, IndexNotReadyError, DocumentFormatError, DocumentNotFoundError, QueryFormatError
from fastapi import FastAPI, HTTPException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI()

@app.post("/resolve-ticket")
async def resolve_ticket(ticket_text: str):
    try:
        rag = RAGPipeline()
        rag.setup_pipeline()
        response = rag.get_relevant_context(ticket_text)
        return response
    except RAGError as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    logging.info("Starting RAG pipeline")
    rag = RAGPipeline()
    rag.setup_pipeline()

if __name__ == "__main__":
    main()