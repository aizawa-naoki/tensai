import csv
import pandas as pd
from utils import get_link_page_title


# Sanitize title for comma-formatted csv
def __sanitize_title(title: str) -> str:
    return title.replace(",", ' ').replace('“', ' ').replace("\"", ' ').replace("\n", ' ')


# Convert titles to single string
def __convert_array_title_to_str(title_array: []) -> str:
    if len(title_array) == 0: return ""

    temp = map(__sanitize_title, title_array)
    return "|".join(temp)


INPUT_FILEPATH = "dataset/train.csv"
OUTPUT_FILEPATH = INPUT_FILEPATH.split(".")[0] + "_title.csv"

data = pd.read_csv(INPUT_FILEPATH)

id_array, text_array = data["id"], data["text"]

title_data = []
for i, (ident, text) in enumerate(zip(id_array, text_array)):
    print(f"{i} - {ident}")
    titles = get_link_page_title(text)
    concat_title = __convert_array_title_to_str(titles)
    title_data.append([str(ident), concat_title])

title_data = pd.DataFrame(title_data)
title_data.columns = ["id", "page_title"]
title_data.to_csv(OUTPUT_FILEPATH, index=False)
