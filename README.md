# InGAN
### Official code for the paper "InGAN: Capturing and Retargeting the DNA of a Natural Image"

Project page: http://www.wisdom.weizmann.ac.il/~vision/ingan/ (See our results and visual comparison to other methods)

**Accepted ICCV'19 (Oral)**
----------
![](/figs/fruits.gif)
----------
If you find our work useful in your research or publication, please cite our work:

```
@InProceedings{InGAN,
  author = {Assaf Shocher and Shai Bagon and Phillip Isola and Michal Irani},
  title = {InGAN: Capturing and Retargeting the "DNA" of a Natural Image},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year = {2019}
}
```
----------

# Usage:
## Test
### Quick example
First you have to [download the example checkpoint file](http://www.wisdom.weizmann.ac.il/~vision/ingan/resource/checkpoint_0075000.pth.tar), and put it in ``` InGAN/examples/fruit/ ```.
Will defaulty run on the fruits image, using an existing checkpoint.
```
python test.py
```
### General testing
See configs.py, for all the options. You can either edit this file or modify configuration from command-line.
Examples:
```
python test.py --input_image_path /path/to/some/image.png  # choose input image
python test.py --test_non_rect  # also output non rectangular transformation results
python test.py --test_vid_scale 2.0, 0.5, 2.5, 0.2  # boundary scales for output video: [max_v, min_v, max_h, min_h]
```
Please see configs.py for many more options


## Train
### Quick example
Will defaulty run on the fruits image.
```
python train.py
```
### General training
See configs.py for all the options. You can either edit this file or modify configuration from command-line.
Examples:
```
python train.py --input_image_path /path/to/some/image.png  # choose input image
python train.py --G_num_resblocks 3  # change number of residual block in the generator
```
Please see configs.py for many more options
### monitoring
In you results folder, monitor files will be periodically created, example:
![](/figs/monitor_60000.png)

## Produce complex animations by scripts:  
Please see the file supp_video.py

## Parallel training for many images
Please see the file train_supp_mat.py
