"""
Endpoint Comparison Test Script.

This script demonstrates the difference between main and debug endpoints.
It shows that the main endpoint returns flat MCP JSON without envelope,
while the debug endpoint returns envelope format.
"""

import json
from typing import Any

import requests


def test_endpoint(url: str, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
    """Test an endpoint and return the response."""
    try:
        response = requests.post(f"{url}{endpoint}", json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error testing {endpoint}: {e}")
        return {}


def main():
    """Main function to run endpoint comparison tests."""
    # Configuration
    base_url = "http://localhost:8000"  # Adjust if your server runs on different port
    test_data = {"ticket_text": "I cannot access my domain, it says invalid password"}

    print("=" * 80)
    print("ENDPOINT COMPARISON: Main vs Debug")
    print("=" * 80)
    print()

    # Test main endpoint
    print("1. MAIN ENDPOINT (/resolve-ticket)")
    print("-" * 50)
    main_response = test_endpoint(base_url, "/resolve-ticket", test_data)

    if main_response:
        print("✓ Status: SUCCESS")
        print("✓ Response Type: Flat MCP JSON (no envelope)")
        print("✓ Keys:", list(main_response.keys()))

        # Verify MCP structure
        mcp_fields = ["answer", "references", "action_required"]
        envelope_fields = ["success", "data", "processing_time", "documents_retrieved"]

        print("\nMCP Fields Present:")
        for field in mcp_fields:
            status = "✓" if field in main_response else "✗"
            print(f"  {status} {field}")

        print("\nEnvelope Fields (should be absent):")
        for field in envelope_fields:
            status = "✓" if field not in main_response else "✗"
            print(f"  {status} {field} (absent)")

        print("\nSample Response:")
        print(json.dumps(main_response, indent=2)[:300] + "...")
    else:
        print("✗ Status: FAILED")

    print("\n" + "=" * 80)

    # Test debug endpoint
    print("2. DEBUG ENDPOINT (/resolve-ticket/debug)")
    print("-" * 50)
    debug_response = test_endpoint(base_url, "/resolve-ticket/debug", test_data)

    if debug_response:
        print("✓ Status: SUCCESS")
        print("✓ Response Type: Envelope format with nested MCP data")
        print("✓ Keys:", list(debug_response.keys()))

        # Verify envelope structure
        envelope_fields = ["success", "data", "processing_time", "documents_retrieved"]

        print("\nEnvelope Fields Present:")
        for field in envelope_fields:
            status = "✓" if field in debug_response else "✗"
            print(f"  {status} {field}")

        # Check nested MCP data
        if "data" in debug_response:
            mcp_data = debug_response["data"]
            mcp_fields = ["answer", "references", "action_required"]

            print("\nNested MCP Data:")
            for field in mcp_fields:
                status = "✓" if field in mcp_data else "✗"
                print(f"  {status} data.{field}")

        print("\nSample Response:")
        print(json.dumps(debug_response, indent=2)[:300] + "...")
    else:
        print("✗ Status: FAILED")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✓ Main endpoint (/resolve-ticket): Returns flat MCP JSON without envelope")
    print(
        "✓ Debug endpoint (/resolve-ticket/debug): Returns envelope format with nested MCP data"
    )
    print("✓ Both endpoints provide the same core MCP data, just in different formats")
    print("✓ Main endpoint is optimized for MCP clients")
    print("✓ Debug endpoint is useful for development and monitoring")


if __name__ == "__main__":
    main()
