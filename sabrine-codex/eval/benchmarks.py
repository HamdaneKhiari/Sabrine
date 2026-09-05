"""
eval/benchmarks.py
Benchmark d'exécution pour Sabrina: Codex.

Contrairement à un jugement à l'œil sur une génération ("ça a l'air faux"),
ce harness génère du code pour une liste de problèmes Python, EXÉCUTE le
résultat, et compare contre des cas de test attendus. Le score (X/N) est
la métrique de référence pour comparer objectivement deux checkpoints
(3750 iters vs 10000 iters, 15M vs 50M, etc.) plutôt que des impressions.

Usage :
    python -m eval.benchmarks
    python -m eval.benchmarks --checkpoint model/checkpoints/sabrina_codex_iter3600.pt
    python -m eval.benchmarks --max_tokens 60 --verbose

Le score est aussi loggé dans eval/logs/bench_log.csv (une ligne par run),
pour tracer la courbe score/itération dans le temps.
"""

import os
import re
import csv
import argparse
import contextlib
import io
from datetime import datetime, timezone

import torch
from tokenizers import Tokenizer

from model.architecture import SabrinaCodex


# ---------------------------------------------------------------------------
# Problèmes : chaque entrée est (nom, prompt_de_depart, fonction_a_extraire, cas_de_test)
#   - prompt_de_depart : ce qu'on donne au modèle pour commencer la génération
#     (signature + ":") — le modèle doit continuer avec le corps de la fonction
#   - fonction_a_extraire : nom de la fonction Python attendue dans le code généré,
#     utilisé pour la récupérer depuis le namespace après exec()
#   - cas_de_test : liste de (args_tuple, valeur_attendue)
#
# Volontairement petit (V0) : 15 problèmes simples, pas pour juger le niveau
# définitif de Sabrina, juste pour avoir un point zéro comparable dans le temps.
# ---------------------------------------------------------------------------
PROBLEMS = [
    {
        "name": "add",
        "prompt": "def add(a, b):\n    ",
        "func": "add",
        "tests": [((2, 3), 5), ((10, 20), 30), ((-1, 1), 0)],
    },
    {
        "name": "subtract",
        "prompt": "def subtract(a, b):\n    ",
        "func": "subtract",
        "tests": [((10, 3), 7), ((0, 5), -5)],
    },
    {
        "name": "multiply",
        "prompt": "def multiply(a, b):\n    ",
        "func": "multiply",
        "tests": [((3, 4), 12), ((0, 9), 0)],
    },
    {
        "name": "is_even",
        "prompt": "def is_even(n):\n    ",
        "func": "is_even",
        "tests": [((2,), True), ((3,), False), ((0,), True)],
    },
    {
        "name": "is_odd",
        "prompt": "def is_odd(n):\n    ",
        "func": "is_odd",
        "tests": [((3,), True), ((4,), False)],
    },
    {
        "name": "factorial",
        "prompt": "def factorial(n):\n    ",
        "func": "factorial",
        "tests": [((0,), 1), ((5,), 120), ((3,), 6)],
    },
    {
        "name": "max_of_two",
        "prompt": "def max_of_two(a, b):\n    ",
        "func": "max_of_two",
        "tests": [((3, 7), 7), ((10, 2), 10)],
    },
    {
        "name": "min_of_two",
        "prompt": "def min_of_two(a, b):\n    ",
        "func": "min_of_two",
        "tests": [((3, 7), 3), ((10, 2), 2)],
    },
    {
        "name": "abs_value",
        "prompt": "def abs_value(n):\n    ",
        "func": "abs_value",
        "tests": [((-5,), 5), ((5,), 5), ((0,), 0)],
    },
    {
        "name": "square",
        "prompt": "def square(n):\n    ",
        "func": "square",
        "tests": [((3,), 9), ((-4,), 16)],
    },
    {
        "name": "reverse_string",
        "prompt": "def reverse_string(s):\n    ",
        "func": "reverse_string",
        "tests": [(("abc",), "cba"), (("",), "")],
    },
    {
        "name": "string_length",
        "prompt": "def string_length(s):\n    ",
        "func": "string_length",
        "tests": [(("hello",), 5), (("",), 0)],
    },
    {
        "name": "list_sum",
        "prompt": "def list_sum(numbers):\n    ",
        "func": "list_sum",
        "tests": [(([1, 2, 3],), 6), (([],), 0)],
    },
    {
        "name": "count_vowels",
        "prompt": "def count_vowels(s):\n    ",
        "func": "count_vowels",
        "tests": [(("hello",), 2), (("bcd",), 0)],
    },
    {
        "name": "fibonacci",
        "prompt": "def fibonacci(n):\n    ",
        "func": "fibonacci",
        "tests": [((0,), 0), ((1,), 1), ((6,), 8)],
    },
]


def load_model(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = SabrinaCodex(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, config, checkpoint.get("iter", "?")


def extract_function_body(prompt: str, generated_text: str) -> str:
    """
    Isole une seule définition de fonction depuis le texte généré.

    Le modèle continue après le prompt (qui contient déjà "def nom(...):\\n    "),
    donc on reconstruit le code complet, puis on coupe au premier signe de
    "nouvelle unité de code" pour ne garder QUE la fonction visée :
      - le marqueur <|endofcode|> (déjà géré par eos_token_id côté generate(),
        mais on le retire ici s'il traîne dans le texte décodé)
      - une nouvelle ligne "def " ou "class " non indentée (le modèle a enchaîné
        sur autre chose)
    """
    full_code = prompt + generated_text
    full_code = full_code.split("<|endofcode|>")[0]

    lines = full_code.split("\n")
    kept = [lines[0]] if lines else []
    for line in lines[1:]:
        # Une ligne non-indentée (hors ligne vide) qui n'est pas la première
        # marque la fin de la fonction courante.
        if line.strip() and not line.startswith((" ", "\t")):
            break
        kept.append(line)

    return "\n".join(kept)


def run_problem(model, tokenizer, problem: dict, device: str, max_tokens: int,
                 eos_token_id, verbose: bool) -> tuple[bool, str]:
    prompt_ids = tokenizer.encode(problem["prompt"]).ids
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_idx = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=1.0,
        top_k=1,  # greedy déterministe : on veut un score reproductible, pas de créativité
        eos_token_id=eos_token_id,
        repetition_penalty=1.2,
    )

    generated_text = tokenizer.decode(output_idx[0, len(prompt_ids):].tolist())
    code = extract_function_body(problem["prompt"], generated_text)

    namespace: dict = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102 — usage local volontaire, contenu = notre propre modèle
        func = namespace.get(problem["func"])
        if func is None:
            return False, f"fonction '{problem['func']}' absente du code généré"

        for args, expected in problem["tests"]:
            try:
                result = func(*args)
            except Exception as e:
                return False, f"exception à l'exécution: {e!r}"
            if result != expected:
                return False, f"{problem['func']}{args} -> {result!r} (attendu {expected!r})"

        return True, "OK"

    except Exception as e:
        return False, f"erreur de syntaxe/exec: {e!r}"
    finally:
        if verbose:
            print(f"\n--- {problem['name']} : code généré ---\n{code}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/sabrina_codex_final.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/sabrina_tokenizer.json")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--verbose", action="store_true", help="Affiche le code généré pour chaque problème")
    parser.add_argument("--log", type=str, default="eval/logs/bench_log.csv")
    args = parser.parse_args()

    device = "cpu"  # l'inférence sur un modèle de cette taille est instantanée en CPU, pas besoin de GPU

    model, config, ckpt_iter = load_model(args.checkpoint, device)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_token_id = tokenizer.token_to_id("<|endofcode|>")

    print(f"[Bench] Checkpoint : {args.checkpoint} (itération {ckpt_iter})")
    print(f"[Bench] {len(PROBLEMS)} problèmes\n")

    results = []
    for problem in PROBLEMS:
        passed, detail = run_problem(
            model, tokenizer, problem, device, args.max_tokens, eos_token_id, args.verbose
        )
        results.append((problem["name"], passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  {problem['name']:<18} {status:<5} {'' if passed else detail}")

    n_passed = sum(1 for _, passed, _ in results if passed)
    print(f"\n[Bench] Score : {n_passed}/{len(PROBLEMS)}")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    write_header = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "checkpoint", "iter", "score", "total"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(), args.checkpoint, ckpt_iter,
            n_passed, len(PROBLEMS),
        ])
    print(f"[Bench] Résultat loggé dans {args.log}")


if __name__ == "__main__":
    main()