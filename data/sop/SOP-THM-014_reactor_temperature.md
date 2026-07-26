---
id: SOP-THM-014 r6
type: sop
title: Reactor Temperature Excursion Response
param: temp
asset: BR-12
effective: 2024-03-11
---

# Purpose

This procedure defines the corrective response when reactor temperature on BR-12 departs the validated control band of 36.5-37.5 degC during the fermentation/growth phase.

# Scope

Applies to all fermentation batches of Omez, Nise and Mintop in Reactor BR-12. Applies to alarm-zone excursions (36.0-38.0 degC). Trip-zone excursions require QA notification before any corrective action.

# Responsibilities

The Production Operator performs first-line response. The QA Reviewer approves any setpoint change that exceeds 5 percent of the nominal coolant setpoint. Engineering supports heat-exchanger interventions.

# Procedure — upward excursion (temperature rising)

Step 1. Confirm the reading against the redundant probe TT-1202B. If the two probes disagree by more than 0.3 degC, treat as probe fault and raise a deviation instead of adjusting the process.

Step 2. Reduce the coolant-loop setpoint in increments of no more than 4 percent. Wait one full residence time (approximately 90 seconds) between increments and observe the trend before further adjustment. Over-correction causes an undershoot that is harder to recover than the original excursion.

Step 3. Verify chiller CW-3 discharge temperature is at or below 8 degC. If discharge temperature is elevated, the chiller is the limiting factor and further coolant setpoint reduction will not help.

Step 4. Read heat-exchanger differential pressure. If heat-exchanger dP exceeds 35 kPa, fouling is indicated. Raise a maintenance work order for heat-exchanger cleaning and continue to hold the batch under enhanced monitoring.

Step 5. If temperature has not returned to the control band within 15 minutes of the first adjustment, escalate to the Production Lead and initiate a time-out-of-spec deviation record.

# Procedure — downward excursion (temperature falling)

Step 1. Confirm the reading against the redundant probe.

Step 2. Verify the steam valve is responding to demand. A stuck steam valve is the most common cause of a downward excursion.

Step 3. Increase the jacket temperature setpoint in increments of no more than 3 percent, observing one residence time between increments.

# Acceptance criteria

The excursion is closed when the parameter has remained within 36.5-37.5 degC for 10 consecutive minutes and the deviation record has been reviewed by QA.
