## Architecture 
In this method, images from a video are passed through a CNN, which extracts feature vectors for each frame. These feature vectors are then fed into an RNN, which processes the sequence and outputs a result using a softmax layer, typical for classification tasks.

A 3D Convolutional Network (Conv3D) extends the concept of 2D convolutions by adding a third dimension. In 2D convolutions, the filter moves in two directions (x and y), while in 3D convolutions, it moves in three directions (x, y, and z). The input to Conv3D is a video, which can be thought of as a sequence of 30 RGB images. If each image has a size of 100x100x3, the video becomes a 4D tensor of size 100x100x30x3, or equivalently, (100x100x30)x3, where 3 represents the number of channels. Just like a 2D convolution uses a (fxf)xc filter (with f being the filter size and c the number of channels), a 3D convolution uses a (fxfxf)xc cubic filter. This cubic filter moves through the three dimensions of the (100x100x30) tensor, convolving across all channels.

## Steps
- Load the training and validation datasets.
- Set the hyperparameters, including batch size and the number of epochs.
- Crop and resize all images.
- Develop a custom generator function to preprocess the images and create batches of video frames, where each video is represented as (number of images, height, width, number of channels).
- Build a Conv3D model using MaxPooling3D layers and a softmax output layer. Ensure the model achieves good accuracy with minimal parameters to fit within webcam memory constraints.
- Design a Conv2D+GRU model using the TimeDistributed layer.
- Additionally, implement a Conv2D+LSTM model.
- Create a transfer learning model combining MobileNet with LSTM.
- Compare the models based on training and validation accuracy, number of trainable parameters, and training time.
