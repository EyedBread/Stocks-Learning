import input


df = input.load_data('data/original_dataset/source_price.csv')

data_train, data_test, _, _, _, _, _ = input.partition_data(df)

print('data_train.shape',data_train.shape)
print('data_test.shape',data_test.shape)
print('data_train[0:5]',data_train[0:5])
print('data_test[0:5]',data_test[0:5])
