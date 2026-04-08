from predict import predict_risk
from agent import agent_decision

def run_demo():
    # Correct feature order
    sample_num = [2, 150, 1, 80, 65, 70, 0]

    sample_text = "Pipeline failed due to timeout and high CPU usage"

    prediction = predict_risk(sample_num, sample_text)
    decision = agent_decision(prediction)

    print("\n🔮 Prediction:", prediction)
    print("🤖 Decision:", decision)

if __name__ == "__main__":
    run_demo()