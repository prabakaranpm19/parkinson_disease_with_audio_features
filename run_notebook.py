import json
import traceback

print("Loading notebook...")
with open("parkinsons_speech_detection.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract code cells
code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
print(f"Found {len(code_cells)} code cells.")

# Global state to execute cells in the same environment
globals_dict = {}
import matplotlib
matplotlib.use('Agg')

for idx, cell in enumerate(code_cells):
    print(f"\n==========================================")
    print(f"Executing Cell {idx + 1}...")
    print(f"==========================================")
    
    # Reconstruct the code string
    code = "".join(cell["source"])
    
    # Strip out pip installs (which might fail or hang if run in exec)
    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("!"):
            print(f"Skipping shell command: {line}")
            cmd = line.strip().lstrip("!")
            import sys
            if cmd.startswith("pip "):
                cmd = f'"{sys.executable}" -m ' + cmd
            print(f"Executing shell command in subprocess: {cmd}")
            import subprocess
            subprocess.run(cmd, shell=True)
        else:
            cleaned_lines.append(line)
            
    cleaned_code = "\n".join(cleaned_lines)
    
    try:
        # Execute the cell code
        exec(cleaned_code, globals_dict)
        print(f"Cell {idx + 1} executed successfully!")
    except Exception as e:
        print(f"ERROR executing Cell {idx + 1}:")
        traceback.print_exc()
        exit(1)

print("\nAll cells executed successfully!")
