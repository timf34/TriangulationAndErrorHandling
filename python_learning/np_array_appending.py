import numpy as np

samp_tuples = [(1188, 419), (531, 323), (202, 398)]

# We want to add these tuples to a numpy array as distinct points -> [[1188, 419], [531, 323], [202, 398]]

# Create an empty np array to store the pixel coordinates as distinct points -> [[1188, 419], [531, 323], [202, 398]]
np_arr = np.array([])

# Add the tuples to the np array
for tup in samp_tuples:
    np_arr = np.append(np_arr, tup, axis=0)

# Now adjust the shape of the array to be 3 rows and 2 columns
np_arr = np_arr.reshape(3, 2)

print(np_arr)

n_tuples = [(0, 0), (0, 64), (0, 10), (16, 10), (16, 54), (0, 23), (5, 23), (5, 41), (0, 41), (0, 29.5), (0, 34.5), (16, 24), (16, 40), (51, 24), (51, 40)]
# Now show how we would reshape the array for n tuples
np_arr = np.array([])
for tup in n_tuples:
    np_arr = np.append(np_arr, tup, axis=0)

# np_arr = np_arr.reshape(len(n_tuples), 2)
# print(np_arr)

# Reshape using np_arr
# TODO: note down this really good copilot example.
np_arr = np_arr.reshape(-1, 2)  # This is the one! -1 means that the number of rows is unknown, but the number of columns is 2
print(np_arr)



