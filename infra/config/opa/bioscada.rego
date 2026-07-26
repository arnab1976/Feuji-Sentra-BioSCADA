# =====================================================================
# BioSCADA AI — Policy Decision Point (Open Policy Agent)
#
# Free replacement for a commercial policy engine. Encodes the GxP rules
# that decide whether an agent may execute a control action autonomously.
#
# Test:  opa test config/opa -v
# Query: curl -X POST localhost:8181/v1/data/bioscada/decision \
#          -d '{"input":{"param":"ph","zone":"alarm","p_breach":0.5,
#                        "roles":["agent.execute.ph"],"action":"setpoint.adjust"}}'
# =====================================================================
package bioscada

import rego.v1

default decision := "DENY"
default allow := false

# ---------------------------------------------------------------- RBAC
required_role := sprintf("agent.execute.%s", [input.param])

has_role if required_role in input.roles

# ------------------------------------------------- e-signature triggers
# 1. pH setpoint changes ALWAYS require QA e-signature (SOP-PH-009 r4)
esign_required if input.param == "ph"

# 2. Trip-zone excursions are high-risk
esign_required if input.zone == "trip"

# 3. High predicted breach probability
esign_required if input.p_breach >= 0.80

# ------------------------------------------------------------ decisions
decision := "REQUIRE_ESIGN" if {
    has_role
    esign_required
}

decision := "ALLOW" if {
    has_role
    not esign_required
}

allow if decision == "ALLOW"

# ---------------------------------------------------------- obligations
obligations contains "log_worm"

obligations contains "require_esign" if esign_required
obligations contains "qa_review_24h" if esign_required
obligations contains "quarantine_review" if input.zone == "trip"

# ------------------------------------------------------------- reasons
reason := "RBAC denied: agent lacks the required execute role" if not has_role
reason := "SOP-PH-009: all pH setpoint changes require QA e-signature" if {
    has_role
    input.param == "ph"
}
reason := "Trip-zone excursion — QA review obligation" if {
    has_role
    input.param != "ph"
    input.zone == "trip"
}
reason := "High predicted breach probability (>= 0.80)" if {
    has_role
    input.param != "ph"
    input.zone != "trip"
    input.p_breach >= 0.80
}
reason := "Micro-adjustment within policy" if {
    has_role
    not esign_required
}
