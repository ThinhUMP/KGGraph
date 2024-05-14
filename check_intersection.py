with open("Data/zinc/all.txt", "r") as f:
    data = [line.strip("\r\n").split()[0] for line in f]
    
with open("Data/zinc/test.txt", "r") as f:
    data_test = [line.strip("\r\n").split()[0] for line in f]
    
# Convert the lists to sets
data_set = set(data)
data_test_set = set(data_test)

# Find the intersection
intersection = data_set.intersection(data_test_set)

# Print the intersection
print(len(intersection))