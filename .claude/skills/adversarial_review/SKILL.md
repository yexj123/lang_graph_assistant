---
name: adversarial-review
description: Try to break the current solution using adversarial, malformed, contradictory, incomplete, and prompt-injection inputs.
---

# Adversarial Review

Review the current system without modifying it.

Test where applicable:

- incomplete inputs
- malformed inputs
- contradictory information
- missing evidence
- irrelevant instructions
- prompt injection
- jailbreak-style instructions
- unexpected labels
- empty retrieval results
- tool failures
- hallucination opportunities

For every failure report:

1. Input/scenario
2. Observed behavior
3. Expected behavior
4. Root cause
5. Severity
6. Recommended fix

Focus on concrete failures rather than hypothetical concerns.