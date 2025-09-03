# Self-Check Guide

This document provides comprehensive self-check scripts for developers to verify the system functionality, performance, and compliance.

## Offline Testing

**All 24 unit/regression tests run offline with mocked services, ensuring stable CI/CD without external dependencies.**

### Run Offline Tests
```bash
cd src
python -m pytest tests/ -v
```

### Test Coverage
- **API Tests**: 8 tests - Endpoint testing with mocked services
- **Golden Cases**: 6 tests - Regression tests for core functionality
- **LLM Tests**: 4 tests - LLM service functionality testing
- **RAG Tests**: 6 tests - RAG pipeline testing
- **Total**: 24 tests - All run offline with mocks

## Quick Self-Check (15 minutes)

### Prerequisites
- Server running on `localhost:8000`
- `curl` and `jq` installed
- OpenAI API key configured

### 1. Start the Server
```bash
# Option 1: Direct uvicorn
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Option 2: Docker Compose
docker-compose --profile dev up --build
```

### 2. Test Input Validation (Should return 422)
```bash
# Empty input
curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' \
  -d '{"ticket_text": ""}' | jq

# Over-length input (>5000 chars)
curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' \
  -d '{"ticket_text": "'$(printf 'A%.0s' {1..5001})'"}' | jq
```

**Expected**: Both should return HTTP 422 with validation error details.

### 3. Test Normal Input (Should return flat MCP JSON)
```bash
curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' \
  -d '{"ticket_text":"My domain was suspended and I didn'\''t get any notice."}' | jq
```

**Expected Response:**
```json
{
  "answer": "Based on the documentation, your domain may have been suspended due to...",
  "references": [
    {
      "doc_id": "doc_001",
      "title": "Domain Suspension Guidelines",
      "section": "Reasons for Suspension",
      "url": null
    }
  ],
  "action_required": "escalate_to_support"
}
```

### 4. JSON Compliance Smoke Test (100 requests)
```bash
# Test JSON schema compliance with 100 requests
echo "Starting JSON compliance smoke test..."
for i in {1..100}; do
  curl -s -X POST :8000/resolve-ticket \
    -H 'Content-Type: application/json' \
    -d '{"ticket_text":"DNS not resolving after nameserver change"}' | jq empty || echo "❌ JSON validation failed at request $i"
done
echo "✅ All 100 requests passed JSON validation"
```

### 5. Debug Endpoint Test
```bash
# Test debug endpoint with envelope format
curl -s -X POST :8000/resolve-ticket/debug -H 'Content-Type: application/json' \
  -d '{"ticket_text":"How to transfer my domain?"}' | jq
```

**Expected Debug Response:**
```json
{
  "success": true,
  "data": {
    "answer": "...",
    "references": [...],
    "action_required": "..."
  },
  "processing_time": 1.234,
  "documents_retrieved": 2
}
```

### 6. Health Check
```bash
curl -s :8000/ | jq
```

**Expected:**
```json
{
  "message": "Ticket Resolution API",
  "version": "1.0.0",
  "status": "healthy",
  "performance": {
    "uptime_seconds": 123.45,
    "total_requests": 0,
    "avg_response_time": 0
  }
}
```

### 7. Metrics Endpoint
```bash
curl -s :8000/metrics | jq
```

## Load Testing

### Basic Load Test (10 concurrent requests)
```bash
#!/bin/bash
echo "Starting basic load test..."

for i in {1..10}; do
  (
    curl -s -X POST :8000/resolve-ticket \
      -H 'Content-Type: application/json' \
      -d '{"ticket_text":"Test request '$i' - domain transfer issue"}' | jq -r '.action_required'
  ) &
done

wait
echo "Load test completed"
```

### Stress Test (50 requests)
```bash
#!/bin/bash
echo "Starting stress test with 50 requests..."

start_time=$(date +%s)
for i in {1..50}; do
  curl -s -X POST :8000/resolve-ticket \
    -H 'Content-Type: application/json' \
    -d '{"ticket_text":"Stress test request '$i' - DNS configuration help"}' > /dev/null &

  # Limit concurrent requests to 10
  if (( i % 10 == 0 )); then
    wait
  fi
done

wait
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Stress test completed in ${duration} seconds"
```

## MCP Compliance Testing

### Test All Action Types
```bash
# Test different ticket types to verify all action_required values
echo "Testing action_required enum compliance..."

# Should return "none"
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"What is DNS?"}' | jq -r '.action_required'

# Should return "escalate_to_support"
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"Complex technical issue not in documentation"}' | jq -r '.action_required'

# Should return "escalate_to_abuse_team"
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"Someone is using my domain for spam"}' | jq -r '.action_required'

# Should return "contact_customer"
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"I need help but not sure what information to provide"}' | jq -r '.action_required'
```

### Test References Structure
```bash
# Verify structured references format
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"Domain suspension help"}' | jq '.references[0]'
```

**Expected:**
```json
{
  "doc_id": "doc_001",
  "title": "Domain Suspension Guidelines",
  "section": "Reasons for Suspension",
  "url": null
}
```

## Error Handling Tests

### Test Service Unavailability
```bash
# Stop the server and test error handling
# (This would require stopping the server first)
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"Test when server is down"}' | jq
```

### Test Invalid JSON
```bash
# Test malformed JSON
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text": "Missing closing quote}' | jq
```

## Performance Benchmarks

### Response Time Test
```bash
#!/bin/bash
echo "Testing response times..."

for i in {1..10}; do
  start_time=$(date +%s.%N)
  curl -s -X POST :8000/resolve-ticket \
    -H 'Content-Type: application/json' \
    -d '{"ticket_text":"Performance test request '$i'"}' > /dev/null
  end_time=$(date +%s.%N)

  duration=$(echo "$end_time - $start_time" | bc)
  echo "Request $i: ${duration}s"
done
```

## Docker Testing

### Test Docker Compose
```bash
# Start with Docker Compose
docker-compose --profile dev up --build

# Test in another terminal
curl -s :8000/ | jq

# Test main endpoint
curl -s -X POST :8000/resolve-ticket \
  -H 'Content-Type: application/json' \
  -d '{"ticket_text":"Docker test"}' | jq
```

### Test Production Profile
```bash
# Test production configuration
docker-compose --profile prod up --build

# Verify production mode
curl -s :8000/ | jq
```

## Automated Test Script

### Complete Self-Check Script
```bash
#!/bin/bash
# complete_self_check.sh

set -e

echo "🚀 Starting Complete Self-Check..."

# Check if server is running
if ! curl -s :8000/ > /dev/null; then
  echo "❌ Server not running. Please start with: uvicorn src.main:app --reload --host 0.0.0.0 --port 8000"
  exit 1
fi

echo "✅ Server is running"

# Test 1: Input validation
echo "📋 Testing input validation..."
empty_response=$(curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' -d '{"ticket_text": ""}')
if echo "$empty_response" | jq -e '.detail' > /dev/null; then
  echo "✅ Empty input validation working"
else
  echo "❌ Empty input validation failed"
fi

# Test 2: Normal request
echo "📋 Testing normal request..."
normal_response=$(curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' -d '{"ticket_text":"Test domain issue"}')
if echo "$normal_response" | jq -e '.answer' > /dev/null; then
  echo "✅ Normal request working"
else
  echo "❌ Normal request failed"
fi

# Test 3: JSON compliance (10 requests)
echo "📋 Testing JSON compliance..."
for i in {1..10}; do
  if ! curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' -d '{"ticket_text":"Compliance test '$i'"}' | jq empty > /dev/null; then
    echo "❌ JSON compliance failed at request $i"
    exit 1
  fi
done
echo "✅ JSON compliance test passed"

# Test 4: Debug endpoint
echo "📋 Testing debug endpoint..."
debug_response=$(curl -s -X POST :8000/resolve-ticket/debug -H 'Content-Type: application/json' -d '{"ticket_text":"Debug test"}')
if echo "$debug_response" | jq -e '.success' > /dev/null; then
  echo "✅ Debug endpoint working"
else
  echo "❌ Debug endpoint failed"
fi

echo "🎉 All self-check tests passed!"
```

## Troubleshooting

### Common Issues

1. **Server not responding**
   ```bash
   # Check if server is running
   curl -s :8000/ || echo "Server not running"

   # Check logs
   # Look for startup errors in the server logs
   ```

2. **OpenAI API errors**
   ```bash
   # Check API key
   echo $OPENAI_API_KEY

   # Test with debug endpoint for more details
   curl -s -X POST :8000/resolve-ticket/debug -H 'Content-Type: application/json' -d '{"ticket_text":"Test"}' | jq
   ```

3. **JSON validation failures**
   ```bash
   # Check the raw response
   curl -s -X POST :8000/resolve-ticket -H 'Content-Type: application/json' -d '{"ticket_text":"Test"}' | jq .
   ```

### Performance Issues

1. **Slow responses**
   - Check OpenAI API rate limits
   - Monitor server logs for errors
   - Use debug endpoint to see processing times

2. **Memory issues**
   - Check Docker memory limits
   - Monitor FAISS index size
   - Restart container if needed

## Success Criteria

✅ **All tests should pass:**
- Input validation returns 422 for invalid inputs
- Normal requests return valid MCP JSON
- JSON compliance test passes 100/100 requests
- Debug endpoint returns envelope format
- Health check returns healthy status
- All action_required values are valid enum values
- References are properly structured
- Response times are reasonable (< 5 seconds)

## Performance Statistics

### Golden Cases Test Results
```bash
# Run golden cases with performance metrics
cd src
python -m pytest tests/golden_cases_test.py -v -s
```

**Expected Performance:**
- **Response Time**: < 2 seconds per request
- **Success Rate**: 100% for all golden cases
- **Memory Usage**: < 500MB for typical requests
- **CPU Usage**: < 50% during normal operation

### Load Test Results
```bash
# Run load test script
./load_test.sh
```

**Benchmark Targets:**
- **Concurrent Requests**: 10 requests/second
- **Response Time**: < 3 seconds (95th percentile)
- **Error Rate**: < 1%
- **Memory Stability**: No memory leaks over 100 requests

### System Metrics
```bash
# Check system performance
curl -s :8000/debug/metrics | jq
```

**Key Metrics:**
- **Uptime**: System availability
- **Total Requests**: Request count
- **Average Response Time**: Performance baseline
- **Success Rate**: Reliability indicator

---

*This self-check guide ensures the system meets all requirements and performs reliably in production.*
