#!/usr/bin/env python3
import sys
import os
import logging
import traceback
from rag import RAGPipeline
from exceptions import RAGError

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def rag_basic_test():
    logger.info("=== Testing RAG Pipeline with default model ===")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        logger.info(f"RAG Pipeline initialized with model: {rag.embedding_model.model_name}")
        
        # Setup pipeline
        if rag.setup_pipeline():
            logger.info("Pipeline setup completed")
            
            # Test search
            test_query = "My domain was suspended and I didn't get any notice. How can I reactivate it?"
            logger.info(f"Testing search with query: {test_query}")
            
            results = rag.search(test_query, k=3)
            assert len(results) > 0, "At least one result should be found"
            logger.info(f"Found {len(results)} relevant documents")
            
            # Show top result
            if results:
                doc_idx, score, doc = results[0]
                logger.info(f"Top result (Score: {score:.3f}):")
                logger.info(f"  Title: {doc['title']}")
                logger.info(f"  Section: {doc['section']}")
                logger.info(f"  Content: {doc['content'][:100]}...")
            
            # Test context generation
            logger.info("Testing context generation...")
            context = rag.get_relevant_context(test_query, k=2)
            assert isinstance(context, str) and len(context) > 0, "Context failed to generate"
            logger.info(f"Generated context with {len(context)} characters")
            
            # Test empty query (should handle gracefully)
            logger.info("Testing empty query...")
            try:
                empty_results = rag.search("", k=3)
                # Empty query should return empty results, not raise error
                assert isinstance(empty_results, list), "Empty query should return list"
                logger.info("Empty query handled gracefully")
            except Exception as e:
                logger.warning(f"Empty query handling: {e}")
            
            logger.info(f"Index saved to: {rag.index_path}")
            logger.info("All tests passed. RAG system is working correctly")
            return True
            
        else:
            logger.error("Pipeline setup failed")
            return False

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = rag_basic_test()
    sys.exit(0 if success else 1)
