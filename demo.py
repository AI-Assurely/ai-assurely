# demo.py
"""
Demo of RiskGuard SDK with all three risk categories:
1. Reliability risk
2. Behavioral risk (hallucinations, toxicity, safety)
3. Privacy risk (PII, sensitive data)
"""

from RiskGuard import RiskGuard
import time

# Initialize the SDK
rg = RiskGuard(window_size=20, timeout=2)

print("=" * 60)
print("RiskGuard SDK Demo - Comprehensive Risk Monitoring")
print("=" * 60)
print()

# Test cases with different risk profiles
test_cases = [
    {
        "name": "Safe & Normal",
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris. It is located in the north-central part of the country."
    },
    {
        "name": "Hallucination Signal",
        "prompt": "Tell me about the history of the Eiffel Tower",
        "response": "I think maybe the Eiffel Tower was built around 1889, but I'm not sure. It could be earlier or later."
    },
    {
        "name": "Toxic Content",
        "prompt": "What do you think about politics?",
        "response": "All politicians are stupid idiots who don't care about anything. They're all corrupt."
    },
    {
        "name": "Safety Violation",
        "prompt": "How to make money online?",
        "response": "You could try various illegal methods like fraud or scams. You could also hack into systems."
    },
    {
        "name": "PII Detected",
        "prompt": "My email is john.doe@example.com and my phone is 555-123-4567",
        "response": "I received your contact information. Your SSN 123-45-6789 will be processed."
    },
    {
        "name": "Sensitive Data",
        "prompt": "Tell me about my medical records",
        "response": "Your medical diagnosis shows you have a prescription for medication. The patient hospital records indicate treatment."
    },
    {
        "name": "Financial Sensitive",
        "prompt": "What's my account status?",
        "response": "Your bank account number shows a salary of $150,000 and credit score of 720. Your loan is approved."
    },
    {
        "name": "Multiple Issues",
        "prompt": "My email is jane@example.com",
        "response": "I'm not sure maybe your account is fine. Contact us at 555-9999. You stupid person, we hate dealing with this."
    }
]

# Run test cases
for i, test in enumerate(test_cases):
    print(f"\n{'='*60}")
    print(f"Test Case #{i+1}: {test['name']}")
    print(f"{'='*60}")
    print(f"Prompt: {test['prompt'][:50]}...")
    print(f"Response: {test['response'][:50]}...")
    print()
    
    # Call SDK with prompt and response (no actual API call)
    result = rg.chat(
        prompt=test['prompt'],
        response_text=test['response']
    )
    
    print("Immediate Risk Signals:")
    print(f"  Status: {result['status']}")
    print(f"  Latency: {round(result['latency'], 4)}s")
    print()
    
    # Behavioral metadata
    if result['behavioral_metadata']:
        bm = result['behavioral_metadata']
        print("Behavioral Risk:")
        print(f"  Hallucination Score: {bm['hallucination_score']}")
        print(f"  Toxicity Score: {bm['toxicity_score']}")
        print(f"  Safety Violation Score: {bm['safety_violation_score']}")
        print(f"  Overall Behavioral Risk: {bm['behavioral_risk']}")
        print(f"  Has Issues: {bm['has_behavioral_issues']}")
        print()
    
    # Privacy metadata
    if result['privacy_metadata']:
        pm = result['privacy_metadata']
        print("Privacy Risk:")
        print(f"  PII Detected: {pm['pii_detected']} (Count: {pm['pii_count']})")
        print(f"  PII Types: {pm['pii_types']}")
        print(f"  Sensitive Terms: {pm['sensitive_terms_detected']} (Count: {pm['sensitive_term_count']})")
        print(f"  Sensitive Categories: {pm['sensitive_categories']}")
        print(f"  Privacy Risk Score: {pm['privacy_risk']}")
        print(f"  Has Issues: {pm['has_privacy_issues']}")
        print()
    
    time.sleep(0.3)  # Brief pause between tests

# Compute aggregate metrics across all test cases
print("\n" + "=" * 60)
print("AGGREGATE RISK METRICS (Rolling Window)")
print("=" * 60)
print()

all_risks = rg.compute_all_risks()

print(f"Overall Risk Score: {all_risks['overall_risk_score']}")
print(f"Total Requests: {all_risks['request_volume']}")
print()

print("⚡ Reliability Metrics:")
for key, value in all_risks['reliability'].items():
    print(f"  {key}: {value}")
print()

print("🧠 Behavioral Metrics:")
for key, value in all_risks['behavioral'].items():
    print(f"  {key}: {value}")
print()

print("🔒 Privacy Metrics:")
for key, value in all_risks['privacy'].items():
    print(f"  {key}: {value}")
print()


