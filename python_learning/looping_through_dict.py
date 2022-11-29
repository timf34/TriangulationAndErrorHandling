"""
    This file will be for practicing looping through a dictionary's values to two keys at the same time.
"""

sample_dict = {"a" : [1, 2, 3], "b": [4, 5, 6]}
keys = list(sample_dict.keys())

for i in range(len(sample_dict[keys[0]])):
    print(sample_dict[keys[0]][i], sample_dict[keys[1]][i])


"""
    This section will be to test how to filter a nest dict for matching keys. And note that the dicts might have
    different lengths and numbers of keys!
"""

sample_dict = {"a": {1: "A", 2: "B", 3: "C", 4: "X", 5: "Y", 6: "Z"}, "b": {1: "D", 2: "E", 3: "F", 5: "Z"}}

keys = list(sample_dict.keys())

# This will be the list of keys that are in both dicts
matching_keys = []

# Loop through the first dict's keys
for key in sample_dict[keys[0]].keys():
    # If the key is in the second dict, add it to the list of matching keys
    if key in sample_dict[keys[1]].keys():
        matching_keys.append(key)


print(matching_keys)  # These are our matching frames





