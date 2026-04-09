from predict import predict_risk
from agent import agent_decision
import json

def run_demo():

    test_cases = [
        {
            "name": "High Failure Case",
            "num": [5, 200, 1, 50, 90, 85, 3],
            "text": "Build failed due to multiple errors and high CPU usage"
        },
        {
            "name": "Security Risk Case",
            "num": [0, 120, 0, 80, 60, 50, 0],
            "text": "Detected vulnerability in dependency package"
        },
        {
            "name": "Performance Issue Case",
            "num": [1, 300, 0, 70, 95, 90, 2],
            "text": "High latency and performance degradation observed"
        },
        {
            "name": "Safe Deployment Case",
            "num": [0, 100, 0, 90, 40, 30, 0],
            "text": "Build successful with all tests passed"
        }
    ]

    results = []

    for case in test_cases:
        prediction = predict_risk(case["num"], case["text"])
        decision = agent_decision(prediction)

        result = {
            "Test Case": case["name"],
            "Prediction": prediction.tolist(),
            "Decision": decision,
            "Status": "FAIL" if "Stop Pipeline" in decision else "PASS"
        }

        results.append(result)

        print("\n==============================")
        print(f" {case['name']}")
        print("Prediction:", prediction)
        print("Decision:", decision)
        print("Status:", result["Status"])

    # Save results to file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n Results saved to results.json")

if __name__ == "__main__":
    run_demo()