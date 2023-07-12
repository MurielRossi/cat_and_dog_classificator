import classificator

data_dir = 'cat_&_dog'
img_dir = 'images.jpg'

trainloader, testloader, train_transforms = classificator.load_ad_transform_data(data_dir)
model = classificator.train_and_test(trainloader, testloader)

print("ciao")
predict_class = classificator.pre_image(img_dir, model, data_dir, train_transforms)

print(predict_class)
