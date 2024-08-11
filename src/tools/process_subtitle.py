import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-f", type=str)
parser.add_argument("--language", "-l", type=str, default="JP")
parser.add_argument("--character_name", "-c", type=str)
args = parser.parse_args()

texts = []
with open(args.file_path, encoding="utf-8") as fp:
    lines = fp.readlines()
    target_lang_lines = [line for line in lines if args.language in line]
    target_character_lines = [
        line for line in target_lang_lines if args.character_name in line
    ]

    for line in target_character_lines:
        texts.append(line.split(",")[-1].rstrip("\n"))

# save as json utf-8
# format of json: {"name": character_name, "dialogue": [{content: dialogue_of_the_character}]}
import json

file_name = args.file_path.split(".")[0]
with open(
    f"{file_name}_{args.language}_{args.character_name}.json", "w", encoding="utf-8"
) as fp:
    json.dump(
        {
            "name": args.character_name,
            "dialogue": [{"content": text} for text in texts],
        },
        fp,
        ensure_ascii=False,
        indent=4,
    )
