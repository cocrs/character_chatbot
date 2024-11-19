import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--folder", "-f", type=str)
parser.add_argument("--language", "-l", type=str, default="JP")
parser.add_argument("--mode", "-m", type=str, default="character", choices=["line", "character"])
parser.add_argument("--character_name", "-c", type=str)
args = parser.parse_args()


files = os.listdir(args.folder)
files = [f for f in files if f.endswith(".cht.ass")]
files = [os.path.join(args.folder, f) for f in files]


for file in files:
    print(f"Processing {file}")
    texts = []
    
    if args.mode == "character":
        with open(file, encoding="utf-8") as fp:
            lines = fp.readlines()
            target_lang_lines = [line for line in lines if args.language in line]
            target_character_lines = [
                line for line in target_lang_lines if args.character_name in line
            ]

            for line in target_character_lines:
                texts.append(line.split(",")[-1].rstrip("\n"))

    elif args.mode == "time":
        with open(file, encoding="utf-8") as fp:
            lines = fp.readlines()
            target_lang_lines = [line for line in lines if args.language in line]
            start_time = int(target_lang_lines[0].split(",")[0].split("-")[0])
            end_time = int(target_lang_lines[-1].split(",")[0].split("-")[1])
            for line in target_lang_lines:
                texts.append(line.split(",")[-1].rstrip("\n"))
    
    # save as json utf-8
    # format of json: {"name": character_name, "dialogue": [{content: dialogue_of_the_character}]}
    file_name = file.split("/")[-1].split(".")[0]
    with open(
        f"src/raw_subtitles/{file_name}_{args.language}_{args.character_name}.json",
        "w",
        encoding="utf-8",
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
