#!/usr/bin/env python3
import sys
import os
from rag import RAGPipeline

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def rag_basic_test():
    print("=== Testing RAG Pipeline with default model ===")
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        print(f"RAG Pipeline initialized with model: {rag.embedding_model.model_name}")
        
        # Setup pipeline
        if rag.setup_pipeline():
            print("Pipeline setup completed")
            
            # Test search
            test_query = "My domain was suspended and I didn't get any notice. How can I reactivate it?"
            print(f"\nTesting search with query: {test_query}")
            
            results = rag.search(test_query, k=3)
            print(f"Found {len(results)} relevant documents")
            
            # Show top result
            if results:
                doc_idx, score, doc = results[0]
                print(f"Top result (Score: {score:.3f}):")
                print(f"  Title: {doc['title']}")
                print(f"  Section: {doc['section']}")
                print(f"  Content: {doc['content'][:100]}...")
            
            # Test context generation
            print(f"\nTesting context generation...")
            context = rag.get_relevant_context(test_query, k=2)
            print(f"Generated context with {len(context)} characters")
            
            print(f"\nIndex saved to: {rag.index_path}")
            print("\nAll tests passed! RAG system is working correctly")
            return True
            
        else:
            print("Pipeline setup failed")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = rag_basic_test()
    sys.exit(0 if success else 1)
