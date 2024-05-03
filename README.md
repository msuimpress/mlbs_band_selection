# Measurement Learning-based Band Selection (MLBS)

The repository for the source code of the MLBS algorithm for hyperspectral band selection.

The source_code address includes the Python files including training code for the proposed model MLBS and three alternative deep convolutional neural network-based band selection models BHCNN, CM-CNN, and MEAC. The other Python files include the codework for separating the band selection module of the models and testing them with an SVM classifier, and finding out the selected MLBS bands.

The implementation of the models was done using the TensorFlow library. The codes for training work standalone. You may run any of them in a machine on the terminal or any IDE with Python as long as you keep them together in the same address with the dataset folder.

Please refer to the paper [Learning-Based Optimization of Hyperspectral Band Selection for Classification](https://www.mdpi.com/2072-4292/15/18/4460)
