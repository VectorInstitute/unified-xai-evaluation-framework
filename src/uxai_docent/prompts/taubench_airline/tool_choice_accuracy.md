### Tool Choice Accuracy

You are evaluating whether each tool call is the correct type of tool for the agent’s immediate intent.

Focus only on tool selection, not arguments or business correctness.

Steps (per tool call):
- Identify the tool used.
- Infer the agent’s intended action at that step.
- Check whether the tool category matches the intended action.
- Check whether the tool is necessary, non-redundant, and allowed in context.

Label “match” if:
- The tool type matches the intended action,
- The tool is necessary or reasonably justified,
- No required prerequisite tool was skipped.

Label “no match” if:
- The tool type is wrong for the intent,
- The tool is premature, redundant, or an unnecessary escalation,
- A required prerequisite step was skipped.

Explanation:
Quote the tool name, state the intended action, and justify why the choice was appropriate or not.
