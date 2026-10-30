"""Pool prediction CSVs and print angular-error metrics."""
import argparse
import csv
import json
from bicuecc.metrics import summarize_angular_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+")
    args = parser.parse_args()
    errors = []
    for path in args.predictions:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            errors.extend(float(row["angular_error"]) for row in csv.DictReader(handle))
    print(json.dumps(summarize_angular_errors(errors), indent=2))


if __name__ == "__main__":
    main()
