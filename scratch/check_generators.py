import re
import os

with open(os.path.join(os.path.dirname(__file__), '..', 'generator', 'presets.py'), 'r', encoding='utf-8') as f:
    text = f.read()

gens = set(re.findall(r'["\'](numpy\.[a-zA-Z0-9_\.]+|faker\.[a-zA-Z0-9_\.]+)["\']', text))
print("All generators found in presets.py:")
for g in sorted(gens):
    print(" -", g)
