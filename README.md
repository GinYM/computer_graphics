#Using Unsurpervised Learning for Visual Mining

##Caffe
I use caffe to train and test. I also give out the caffe in directory `ext`

##Model
I have a pretrained model for CNN in directory `model`, which extract the fc6 feature in any image. Using this feature can figure out the similarity of two different image by their normalized correlation.

##Executation
First, you should change `image_dir` in **test.py** to the directory where you store your image database.

Then change the contents in  **name_list.txt** to the name of image in your image database. Also change the contents in **target_name.txt** to the name of image to the target image. The target image suggests that you want to find some image in name_list.txt that is similar to the target_image.

In `output.txt`, all the similarity and their corresponding image name will be shown.

##To do
-  Find the relative position of two patch in the same image.
