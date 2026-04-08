from predict import predict_risk
from agent import agent_decision

def run_demo():
    # Example input
    sample_num = [120, 3, 75, 60, 2]
    sample_text = "Build failed due to dependency error"

    prediction = predict_risk(sample_num, sample_text)
    decision = agent_decision(prediction)

    print("Prediction:", prediction)
    print("Decision:", decision)

if __name__ == "__main__":
    run_demo()