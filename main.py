import classificator

data_dir = 'venv/cats_and_dogs'


trainloader, testloader = classificator.load_ad_transform_data(data_dir)
model = classificator.train_and_test(trainloader, testloader)

