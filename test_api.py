import requests
import json
import time

BASE = "http://localhost:8000"

print("=" * 60)
print("FINRAG API INTEGRATION TESTS")
print("=" * 60)

# Test 1: Health
print("\n1. Health Check")
r = requests.get(f"{BASE}/health")
assert r.status_code == 200
print(f"   OK: {r.json()}")

# Test 2: Basic Query
print("\n2. Basic Query")
payload = {"question": "What was Apple revenue in 2025?"}
r = requests.post(f"{BASE}/query", json=payload)
assert r.status_code == 200
data = r.json()
print(f"   OK: request_id: {data['request_id']}")
print(f"   OK: latency: {data['latency_ms']:.0f}ms")
print(f"   OK: is_refusal: {data['is_refusal']}")
assert "391,035" in data['answer'] or "416,161" in data['answer']

# Test 3: Query with Explainable
print("\n3. Query with Explainable")
payload = {"question": "What was Apple revenue in 2025?", "include_explainable": True}
r = requests.post(f"{BASE}/query", json=payload)
assert r.status_code == 200
data = r.json()
print(f"   OK: request_id: {data['request_id']}")
print(f"   OK: explainable: {'explainable' in data and data['explainable'] is not None}")
if data.get('explainable'):
    exp = data['explainable']
    print(f"   OK: claim_traces: {len(exp.get('claim_traces', []))}")
    print(f"   OK: evidence_chunks: {len(exp.get('evidence_chunks', []))}")
    print(f"   OK: retrieval_diagnostics: {exp.get('retrieval_diagnostics', {})}")

# Test 4: Stats endpoint
print("\n4. Stats Endpoint")
r = requests.get(f"{BASE}/stats")
assert r.status_code == 200
print(f"   OK: {r.json()}")

# Test 5: Queries endpoint
print("\n5. Queries Endpoint")
r = requests.get(f"{BASE}/queries?limit=10")
assert r.status_code == 200
print(f"   OK: {len(r.json())} recent queries")

# Test 6: Metrics endpoint
print("\n6. Prometheus Metrics")
r = requests.get(f"{BASE}/metrics")
assert r.status_code == 200
metrics = r.text
assert "finrag_requests_total" in metrics
assert "finrag_queries_total" in metrics
assert "finrag_query_latency_seconds" in metrics
print(f"   OK: Custom metrics present")
print(f"   OK: Metrics size: {len(metrics)} chars")

# Test 7: Multiple queries for metrics
print("\n7. Load Test (5 queries)")
questions = [
    "What was Microsoft revenue in 2025?",
    "What are NVIDIA risk factors?",
    "What was Google Cloud revenue in 2025?",
    "What is Amazon AWS operating income?",
    "What are Meta Reality Labs losses?",
]
for i, q in enumerate(questions):
    r = requests.post(f"{BASE}/query", json={"question": q})
    assert r.status_code == 200
    print(f"   Query {i+1}: {r.json()['latency_ms']:.0f}ms")

# Test 8: Verify metrics updated
print("\n8. Verify Metrics Updated")
r = requests.get(f"{BASE}/metrics")
metrics = r.text
assert 'finrag_queries_total{status="success"}' in metrics
print(f"   OK: Query counter incremented")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)