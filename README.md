# Toy Project - Deck Recognition
## Intro.

This project is a ordinary practice of mine for:
- Turn my own images into MNIST-like file
- Using the structure of MNIST model to train and predict the card images
- ... etc

My own main scripts in this repo. (under ./src folder):
- conv_cards.py
- conv_cards_inference.py
- [jpg_to_mnist/](https://www.researchgate.net/post/How_to_create_MNIST_type_database_from_images)

Other scripts come from [yeephycho's repo.](http://yeephycho.github.io/2016/08/15/image-data-in-tensorflow/)

## Dataset

The images come from [duel links site](http://duellinks.gamea.co/c/yjdptqt2). *Duel links* is a Collectable Card Game
released on mobile, inspired by Japan manga *Yu-Gi-Oh*.

One of my goals is using CNN to recognize the card. Such as the types of the card (monster, spell or trap), the rarity ... eventually the specific name of it.


## Usage

### Prepare the image input

Note that in jpg_to_mnist/ there are two folders: testing_images & training_images. I put the original jpg files in it, and categorize them into different folders, according to their classes. The class labels are represented by digits, 0 is monster card, 1 is spell card, 2 is trap card. Use resize-script.sh to resize each of them into 28\*28 image.
```bash
cd ../tensorflow_input_image_by_tfrecord/src/jpg_mnist/
./resize-scripts.sh
```
Then run convert-images-to-mnist-format.py to convert them.
```bash
python convert-images-to-mnist-format.py
```
4 files will then occur in the current folders.

### Train with dataset

Under src/
``` bash
python conv_cards.py
```
The learned model will be saved into ./cards_model/, in the name of cards_model.ckpt

### 

```bash
python conv_cards_inference.py --image_path $PATH_TO_YOUR_DIGIT_IMAGE 
# e.g. python conv_mnist_inference.py --image_path=../num2.jpg
```

## Things to be done

- Improve the model (the result might not be very good, since this is not a very complicated problem, nearly 100% is expected ...)
- Construct model evaluation
- ... etc
