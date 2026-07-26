package bioscada_test

import rego.v1
import data.bioscada

# pH always requires e-signature, even in a mild alarm
test_ph_always_requires_esign if {
    bioscada.decision == "REQUIRE_ESIGN" with input as {
        "param": "ph", "zone": "alarm", "p_breach": 0.4,
        "roles": ["agent.execute.ph"], "action": "setpoint.adjust"}
}

# A small temperature deviation may proceed autonomously
test_temp_low_risk_allowed if {
    bioscada.decision == "ALLOW" with input as {
        "param": "temp", "zone": "alarm", "p_breach": 0.5,
        "roles": ["agent.execute.temp"], "action": "setpoint.adjust"}
}

# Trip zone escalates regardless of parameter
test_trip_requires_esign if {
    bioscada.decision == "REQUIRE_ESIGN" with input as {
        "param": "temp", "zone": "trip", "p_breach": 0.5,
        "roles": ["agent.execute.temp"], "action": "setpoint.adjust"}
}

# High probability escalates
test_high_probability_requires_esign if {
    bioscada.decision == "REQUIRE_ESIGN" with input as {
        "param": "cond", "zone": "alarm", "p_breach": 0.91,
        "roles": ["agent.execute.cond"], "action": "setpoint.adjust"}
}

# Wrong agent is denied — the RBAC boundary
test_wrong_agent_denied if {
    bioscada.decision == "DENY" with input as {
        "param": "temp", "zone": "alarm", "p_breach": 0.5,
        "roles": ["agent.execute.ph"], "action": "setpoint.adjust"}
}
