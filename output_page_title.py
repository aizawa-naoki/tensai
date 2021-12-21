import csv

import pandas as pd

from utils import get_link_page_title


# Sanitize title for comma-formatted csv
def __sanitize_title(input: str) -> str:
    return input.replace(",", '').replace('“', '').replace("\"", '').replace("\n", '')


# Convert titles to single string
def __convert_array_title_to_str(input: []) -> str:
    if len(input) == 0:
        return ""

    temp = input[:]
    temp = map(__sanitize_title, temp)

    return "|".join(temp)


# sample_tweets = ["@nxwestmidlands huge \" , \n fire at Wholesale markets ablaze http://t.co/rwzbFVNXER as"]

# removed = replace_http_tag(sample_tweet, "[URL]")
# sanitized = __sanitize_title(sample_tweet)
# sanitized = __convert_array_title_to_str(sample_tweets)
# print(sanitized)


INPUT_FILEPATH = "dataset/train.csv"
OUTPUT_FILEPATH = INPUT_FILEPATH.split(".")[0] + "_title.csv"

train_data = pd.read_csv(INPUT_FILEPATH)

id_array = train_data["id"].values

text_array = train_data["text"]


with open(OUTPUT_FILEPATH, 'a+', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(["id", "page_title"])

    for i, (ident, text) in enumerate(zip(id_array, text_array)):
        print(f"{i} - {ident}")
        titles = get_link_page_title(text)
        concat_title = __convert_array_title_to_str(titles)
        writer.writerow([str(ident), concat_title])


