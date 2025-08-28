"""
    Testing code for different neural network configurations.
    Adapted for Python 3.5.2

    Usage in shell:
        python3.5 test.py

    Network (network.py and network2.py) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Michał Dobrzański, 2016
        dobrzanski.michal.daniel@gmail.com
"""
#librerias a utilizar 
import mnist_loader
import network
import pickle

#Parte del codigo que permite importar el archivo de imagenes para entrenar
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
#Damos el mensaje que se cargaron los datos
print(" Datos cargados correctamente")
'''
# ---------------------
# - network.py example:
#import network

'''
#Parte del codigo que permite entrenar a la red

net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
#guardamos nuestra red
with open("red_prueba.pkl", "wb") as f:
    pickle.dump(net, f)
#damos el mensaje que despues de todas las epocas la red ha sido entrenada
print("Entrenamiento terminado y red guardada en red_prueba.pkl")
