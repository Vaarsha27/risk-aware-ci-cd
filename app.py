from predict import predict_risk
from agent import agent_decision

def run_demo():
    sample_num = [2, 150, 1, 80, 65, 70, 0]
    sample_text = "Pipeline failed due to timeout and high CPU usage"

    prediction = predict_risk(sample_num, sample_text)
    decision = agent_decision(prediction)

    print("Prediction:", prediction)
    print("Decision:", decision)

    # 🔥 IMPORTANT: Print keyword for pipeline control
    if "Stop Pipeline" in decision:
        print("STATUS: FAIL")
    else:
        print("STATUS: PASS")

if __name__ == "__main__":
    run_demo()