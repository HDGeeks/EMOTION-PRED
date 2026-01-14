import json
import sys

ALLOWED_ASPECTS = {
    "miscellaneous", "ambience", "food", "service",
    "staff", "menu", "price", "place"
}

ALLOWED_POLARITIES = {"positive", "negative", "neutral"}

ALLOWED_EMOTIONS = {
    "neutral", "mentioned_only", "satisfaction", "admiration",
    "disappointment", "annoyance", "disgust",
    "gratitude", "mixed_emotions"
}


def validate_absa_file(path: str):
    total_lines = 0
    total_labels = 0
    bad_labels = 0

    with open("edited_300_sample_13_jan.jsonl", "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue

            total_lines += 1

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                bad_labels += 1
                print(f"[JSON ERROR] line {line_no}: {e}")
                continue

            outputs = obj.get("output", [])
            if not isinstance(outputs, list):
                bad_labels += 1
                print(f"[FORMAT ERROR] line {line_no}: output is not a list")
                continue

            for label_no, lab in enumerate(outputs, start=1):
                total_labels += 1
                a = lab.get("aspect")
                p = lab.get("polarity")
                e = lab.get("emotion")

                errors = []
                if a not in ALLOWED_ASPECTS:
                    errors.append(f"aspect={a}")
                if p not in ALLOWED_POLARITIES:
                    errors.append(f"polarity={p}")
                if e not in ALLOWED_EMOTIONS:
                    errors.append(f"emotion={e}")

                if errors:
                    bad_labels += 1
                    snippet = obj.get("input", "")[:80]
                    print(
                        f"[LABEL ERROR] line {line_no}, label {label_no}: "
                        + ", ".join(errors)
                        + f" | text='{snippet}'"
                    )

    print("\n===== SUMMARY =====")
    print(f"Lines checked : {total_lines}")
    print(f"Labels checked: {total_labels}")
    print(f"Bad labels    : {bad_labels}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_file.py <data.jsonl>")
        sys.exit(1)

    validate_absa_file(sys.argv[1])