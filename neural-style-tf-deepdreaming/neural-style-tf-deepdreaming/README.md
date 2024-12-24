---
title: neural-style-tf-deepdreaming
emoji: 🖼️↔️🖼️🤖
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: "4.12.0"
app_file: app.py
pinned: false
---

# Combining Images with Deep Dreaming and Style Transfer

This project is an experimental application v1.0 that combines two images using Deep Dreaming and Style Transfer techniques to create unique and visually appealing results.

## Project Overview

This application allows users to upload two images and adjust parameters to combine them using Deep Dreaming (a technique that amplifies patterns in images using a convolutional neural network) and Style Transfer (which applies the style of one image to another using a neural network). The interface provides sliders to adjust the weight of each image, the density of the style, and the sharpness of the content image.

## Technical Details

The project utilizes the following technologies:
- **Neural Style Transfer**: Implemented using TensorFlow and a pre-trained model from TensorFlow Hub.
- **Deep Dreaming**: Utilizes a pre-trained VGG19 model from PyTorch to extract and amplify image features.
- **Gradio**: Provides an interactive web interface for users to upload images and adjust parameters.

## License

ecl

## Developer Information

Developed by Ramon Mayor Martins, Ph.D. (2024)
- Email: rmayormartins@gmail.com
- Homepage: https://rmayormartins.github.io/
- Twitter: @rmayormartins
- GitHub: https://github.com/rmayormartins

## Acknowledgements

Special thanks to Instituto Federal de Santa Catarina (Federal Institute of Santa Catarina) IFSC-São José-Brazil.

## Contact

For any queries or suggestions, please contact the developer using the information provided above.
