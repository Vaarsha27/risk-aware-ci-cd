def agent_decision(prediction):
    build, delay, security = prediction[0]

    if build == 1:
        return "🚫 Stop pipeline & notify developer"

    elif security == 1:
        return "🔐 Trigger security scan"

    elif delay == 1:
        return "⚡ Allocate more resources"

    else:
        return "✅ Proceed deployment"