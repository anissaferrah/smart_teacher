"""Test if mistral follows the math notation rules in the system prompt.

Sends a slide content with realistic math notation (subscripts, sums,
Greek letters) and checks whether the model verbalizes them in
natural language as instructed, or leaks raw LaTeX/symbols.
"""
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

slide_content = """Weighted mean: assigns different importance (weights) to each value.
xi : value, and wi : weight of that value
Example: A student grades: Homework (weight 20%): 80; Test (weight 30%): 70
Weighted Mean = (Σ wi * xi) / (Σ wi) = (0.2*80 + 0.3*70 + 0.5*90) / (0.2+0.3+0.5) = 82
Some observations contribute more to the weighted average than others."""

system_prompt = """You are an experienced data analysis professor presenting a lecture to M2 students.
You receive the current slide content and must PRESENT only this slide out loud.

ABSOLUTE RULES:
- Focus only on the current slide.
- Start naturally: 'In this section, we will look at...'.
- NEVER read the text word for word. Rephrase.
- ZERO markdown.
- 3 to 5 natural sentences. Only in English.

MATH NOTATION RULES (you are speaking, not writing):
Convert ALL math notation to natural spoken English in your output:
- 'x_i' or 'xi' becomes 'x sub i'; 'x^2' becomes 'x squared'; 'x^n' becomes 'x to the power of n'.
- 'x/y' becomes 'x over y' or 'x divided by y'; 'sqrt(x)' becomes 'the square root of x'.
- 'Sigma x_i' becomes 'the sum of x sub i'.
- 'wi : weight' becomes 'w sub i for the weight', NOT 'wi colon weight'.
- Greek letters: spell their name ('alpha', 'sigma').
- NEVER produce raw LaTeX, dollar signs, backslashes, carets, underscores, curly braces."""

prompt = system_prompt + "\n\nSLIDE CONTENT:\n" + slide_content

print("=== Test : mistral suit-il les regles math ? ===")
t0 = time.time()
resp = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 400},
    },
    timeout=300,
)
elapsed = time.time() - t0
output = resp.json().get("response", "").strip()
print(f"Duree: {elapsed:.1f}s")
print("Sortie:")
print(output)
print()

# Vérifs
checks = [
    ("pas de underscore brut au milieu d'un mot", not any(
        c1.isalpha() and c2.isalpha() and c3 == "_"
        for c1, c2, c3 in zip(output[:-2], output[1:-1], output[2:])
    )),
    ("pas de caret ^", "^" not in output),
    ("pas de dollar $", "$" not in output),
    ("pas de Sigma symbole", "Σ" not in output),
    ("pas de backslash \\", "\\" not in output),
    ("pas de accolades {}", "{" not in output and "}" not in output),
    ("a mentionne 'sub' ou 'subscript' ou 'indice'", any(
        kw in output.lower() for kw in ("sub i", "subscript", "indice", "index")
    )),
]
print("=== Evaluation ===")
for label, passed in checks:
    print(f"  {'OK  ' if passed else 'FAIL'} {label}")
