#cargamos os datos
import sys
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

#plt.ion()   # interactive mode

#gardamos o archivo do cliente na carpeta obxetivo
from PIL import Image  
import PIL  
  
image_path =  sys.argv[1]
im1 = Image.open(image_path)  
im2 = im1.save("C:/Users/Sergio/Desktop/TFG/app/targets/imagenes/image.jpg") 
im3 = im1.save("C:/Users/Sergio/Desktop/TFG/app/public/image.jpg") 


#rede
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['targets']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('É unha {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return 'É unha {}'.format(class_names[preds[j]])
        model.train(mode=was_training)

def imshow(inp, title=None):

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

#modelo preentrenado
modelo = torch.load('./modelo.h5')

#función para preparar as imaxes
transformar = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


data_dir="C:/Users/Sergio/Desktop/TFG/app"

images = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transformar)
                  for x in ['targets']}

dataloaders = {x: torch.utils.data.DataLoader(images[x], batch_size=1,
                                             shuffle=False, num_workers=0)
              for x in ['targets']}

inputs, outro = next(iter(dataloaders['targets']))

class_names = ['formiga', 'abella', 'velutina']


device = torch.device("cpu")
images_so_far = 0
with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['targets']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = modelo(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                print('Especie: {}'.format(class_names[preds[j]]))
