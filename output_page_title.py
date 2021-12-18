import csv
import socket

import numpy as np
import pandas as pd

from tensai.utils import get_link_page_title, replace_http_tag


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


INPUT_FILEPATH = "./dataset/train.csv"
OUTPUT_FILEPATH = "./dataset/page_title.csv"

train_data = pd.read_csv(INPUT_FILEPATH)
title_data = pd.read_csv(OUTPUT_FILEPATH, encoding="utf-16")

id_array = train_data["id"].values

text_array = train_data["text"]

with open(OUTPUT_FILEPATH, 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # writer.writerow(["id", "page_title"])

    for i in range(4827, len(id_array)):
        print(str(i) + " - " + str(id_array[i]))
        titles = get_link_page_title(text_array[i])
        concat_title = __convert_array_title_to_str(titles)
        try:
            writer.writerow([str(id_array[i]), concat_title])
        except UnicodeEncodeError:
            writer.writerow([str(id_array[i]), ""])


