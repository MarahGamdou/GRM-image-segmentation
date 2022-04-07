# GRM-project

In the scope of our GRM course, we explored two s-t graph cut algorithms that use min-cut for image segmentation. The purpose of the segmentation is to separate the object from the background. Our work is based on two parts : transforming the image into a suitable graph for a max-flow based background-foreground segmentation, testing both algorithms on Oxford-IIIT Pet Dataset and comparing their efficiency on the latter task. 


In this repository, we made available the code used to perform the background-foreground segmentation. 

# Getting Started 


In order to run the segmentation, you can use this command : 
```
python interactive_segmentation.py image_name -a algorithm_name -s dowscaled_size 
```

Here are some examples : 
```
python interactive_segmentation.py Abyssinian_82.jpg -a Ford-Fulkerson -s 10
```

```
python interactive_segmentation.py Abyssinian_82.jpg -a Push-Relabel -s 25
```



# Dataset 

the images used in this project are taken from Oxford-IIIT Pet Dataset dataset : https://www.robots.ox.ac.uk/~vgg/data/pets/

