import os
import json

folder = "C:\\Users\\anshu\\Desktop\\Simulation_Project\\Formations"

files = os.listdir(folder)
for i in files:
    if "_" in i:
        formation = f'{"-".join(i.split("_")[0])}_{i.split(".json")[0].split("_")[1]}'
    else:
        formation = "-".join(i.split(".json")[0])
    print(formation)

    new_metadata = {
        "metadata": {
            "formation_home": formation,
            "formation_away": formation,
            "attacking_team": "Home",
            "style": ""
        }
    }

    if i.endswith(".json"):
        file_path = os.path.join(folder, i)
        with open(file_path, 'r') as f:
            content = json.load(f)
        # Remove any existing metadata dict
        content = [entry for entry in content if not (isinstance(entry, dict) and "metadata" in entry)]
        # Add new metadata at the end
        content.append(new_metadata)
        with open(file_path, 'w') as f:
            json.dump(content, f, indent=2)
        print(f"Replaced/added metadata in {file_path}")