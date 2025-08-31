#!/usr/bin/env python3
import sys
import os
import logging
import traceback

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm import LLMService, MCPResponse
from exceptions import LLMProviderError, MCPOutputError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Test core LLM functionality: initialization, generation, and MCP response
def test_llm_core_functionality():
    logger.info("=== Testing Core LLM Functionality ===")
    
    try:
        # LLM service initialization
        llm_service = LLMService()
        logger.info("LLMService initialized successfully")
        
        # Basic message building
        ticket_text = "My domain was suspended. How can I reactivate it?"
        context_docs = [
            {
                "title": "Domain Suspension Guidelines",
                "section": "Reactivation Process",
                "content": "To reactivate a suspended domain, update your WHOIS information and contact support.",
                "ref": "Domain Guidelines - Reactivation (#1)"
            }
        ]
        
        system_message, user_message = llm_service.build_messages(ticket_text, context_docs)
        logger.info("Message building working")
        
        # LLM generation and MCP response
        logger.info("Testing LLM generation with context...")
        structured_response = llm_service.generate_response(ticket_text, context_docs)
        
        # Verify MCP response structure
        assert isinstance(structured_response, MCPResponse)
        assert structured_response.answer
        assert structured_response.references
        assert structured_response.action_required in ["none", "escalate_to_support", "escalate_to_abuse_team", "contact_customer"]
        
        logger.info(f"LLM generation successful:")
        logger.info(f"   Answer: {structured_response.answer[:100]}...")
        logger.info(f"   References: {structured_response.references}")
        logger.info(f"   Action: {structured_response.action_required}")
        
        logger.info("Core LLM functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Core LLM functionality test failed: {e}")
        traceback.print_exc()
        return False

def run_llm_tests():
    logger.info("Starting LLM Core Functionality Tests")
    
    tests = [
        ("Core LLM Functionality", test_llm_core_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"{test_name}: PASSED")
            else:
                logger.error(f"{test_name}: FAILED")
        except Exception as e:
            logger.error(f"{test_name}: ERROR - {e}")
            traceback.print_exc()
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("Core functionality is working correctly.")
        return True
    else:
        logger.error(f"{total - passed} test(s) failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = run_llm_tests()
    sys.exit(0 if success else 1)
