import json
from pathlib import Path


def mine_failure_cases(predictions, output_path):
    failure_cases = []
    for example_index, prediction in enumerate(predictions):
        token_errors = []
        for token, gold_tag, predicted_tag in zip(
            prediction["tokens"],
            prediction["gold_tags"],
            prediction["predicted_tags"],
        ):
            if gold_tag != predicted_tag:
                token_errors.append(
                    {
                        "token": token,
                        "gold_tag": gold_tag,
                        "predicted_tag": predicted_tag,
                    }
                )

        if token_errors:
            failure_cases.append(
                {
                    "example_index": example_index,
                    "tokens": prediction["tokens"],
                    "errors": token_errors,
                }
            )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for failure_case in failure_cases:
            file.write(json.dumps(failure_case) + "\n")

    return failure_cases
