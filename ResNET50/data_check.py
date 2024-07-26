from data_preparation import prepare_pairs

# Path to the extracted SOCOFing dataset
data_path = './SOCOFing/Real/'

# Call the function to prepare pairs and print the number of pairs
train_pairs, test_pairs = prepare_pairs(data_path)