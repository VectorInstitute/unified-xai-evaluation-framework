
import os
import json
import importlib.util
import weave
import traceback

try:
    # Initialize weave
    weave.init("gaia_debug_1770845006")
    
    # Load input data
    with open("input.json", "r") as f:
        input_data = json.load(f)
    
    # Load agent arguments
    with open("agent_args.json", "r") as f:
        agent_args = json.load(f)

    # Import agent module
    spec = importlib.util.spec_from_file_location(
        "main",
        os.path.join(os.getcwd(), "main.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_fn = getattr(module, "run")
    
    # Run the agent function
    with weave.attributes({"weave_task_id": "f46b4380-207e-4434-820b-f32ce04ae2a4"}):
        result = agent_fn(input_data, **agent_args)
    
    # Save output
    with open("output.json", "w") as f:
        json.dump(result, f)

except Exception as e:
    print(f"Error running agent: {e}")
    print(traceback.format_exc())
    with open("error.log", "w") as f:
        f.write(f"ERROR: {str(e)}\n")
        f.write(traceback.format_exc())
    raise
