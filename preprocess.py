import os

data_path = "data"

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith(".mid"):
            print(os.path.join(root, file))