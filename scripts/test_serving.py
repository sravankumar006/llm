"""
Automated verification script for CommandLLM FastAPI CPU serving.
Sends requests to http://127.0.0.1:8000 and prints responses and latency metrics.
"""

import sys
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"

def test_serving():
    client = httpx.Client(base_url=BASE_URL, timeout=30.0)

    print("=" * 72)
    print(" 1. Verifying /health Endpoint")
    print("=" * 72)
    resp = client.get("/health")
    print(f"Status Code: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200, f"Health check failed: {resp.text}"

    print("\n" + "=" * 72)
    print(" 2. Testing Linux Command Generation")
    print("=" * 72)
    payload_linux = {
        "os": "linux",
        "prompt": "kill process listening on port 8080",
        "temperature": 0.2,
        "top_k": 40
    }
    resp_linux = client.post("/api/generate", json=payload_linux)
    print(f"Status Code: {resp_linux.status_code}")
    data_linux = resp_linux.json()
    print(f"OS        : {data_linux.get('os')}")
    print(f"Prompt    : {data_linux.get('prompt')}")
    print(f"Command   : {data_linux.get('command')}")
    print(f"Latency   : {data_linux.get('latency_ms')} ms")
    print(f"Raw Tokens: {data_linux.get('raw_tokens')}")
    assert resp_linux.status_code == 200

    print("\n" + "=" * 72)
    print(" 3. Testing PowerShell Command Generation")
    print("=" * 72)
    payload_ps = {
        "os": "powershell",
        "prompt": "kill process listening on port 8080",
        "temperature": 0.2,
        "top_k": 40
    }
    resp_ps = client.post("/api/generate", json=payload_ps)
    print(f"Status Code: {resp_ps.status_code}")
    data_ps = resp_ps.json()
    print(f"OS        : {data_ps.get('os')}")
    print(f"Prompt    : {data_ps.get('prompt')}")
    print(f"Command   : {data_ps.get('command')}")
    print(f"Latency   : {data_ps.get('latency_ms')} ms")
    print(f"Raw Tokens: {data_ps.get('raw_tokens')}")
    assert resp_ps.status_code == 200

    print("\n" + "=" * 72)
    print(" 4. Testing Dual-Mode Comparison Endpoint (/api/compare)")
    print("=" * 72)
    payload_compare = {
        "prompt": "kill process listening on port 8080",
        "temperature": 0.2,
        "top_k": 40
    }
    resp_comp = client.post("/api/compare", json=payload_compare)
    print(f"Status Code: {resp_comp.status_code}")
    data_comp = resp_comp.json()
    print(f"Prompt        : {data_comp.get('prompt')}")
    print(f"Linux Cmd     : {data_comp.get('linux')}")
    print(f"PowerShell Cmd: {data_comp.get('powershell')}")
    print(f"Total Latency : {data_comp.get('latency_ms')} ms")
    assert resp_comp.status_code == 200

    print("\n" + "=" * 72)
    print(" 5. Testing Secondary Query via /api/compare")
    print("=" * 72)
    payload_compare_2 = {
        "prompt": "find all files larger than 100MB",
        "temperature": 0.2,
        "top_k": 40
    }
    resp_comp_2 = client.post("/api/compare", json=payload_compare_2)
    print(f"Status Code: {resp_comp_2.status_code}")
    data_comp_2 = resp_comp_2.json()
    print(f"Prompt        : {data_comp_2.get('prompt')}")
    print(f"Linux Cmd     : {data_comp_2.get('linux')}")
    print(f"PowerShell Cmd: {data_comp_2.get('powershell')}")
    print(f"Total Latency : {data_comp_2.get('latency_ms')} ms")
    assert resp_comp_2.status_code == 200

    print("\n" + "=" * 72)
    print(" [SUCCESS] All API tests passed! CPU Inference functioning correctly.")
    print("=" * 72)

if __name__ == "__main__":
    test_serving()
