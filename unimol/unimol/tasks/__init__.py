from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
# for file in sorted(Path(__file__).parent.glob("*.py")):
#     if not file.name.startswith("_"):
#         importlib.import_module("unimol.tasks." + file.name[:-3])
# importlib.import_module("unimol.tasks.drugfewshot_cross")
# importlib.import_module("unimol.tasks.drugfewshot_layerad")
importlib.import_module("unimol.tasks.drugfewshot_adpater")
# importlib.import_module("unimol.tasks.drugclip")
# importlib.import_module("unimol.tasks.drugclipcross")