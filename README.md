## CALC2.0

Convolutional Autoencoder for Loop Closure 2.0.

To get started, download the COCO dataset and the "stuff" annotations, then run `dataset/gen_tfrecords.py`.
Make sure to unzip the tar in the dataset directory first.
Doing this will generate the sharded tfrecord files as well as `loss_weights.txt`.

After that you can train with `calc2.py`.

Check the --mode options in calc2.py to see what else you can do, like PR curves and finding the best model in a directory.

If you use this code for your research, please cite our paper:
```
@InProceedings{Merrill2019IROS,
  Title                    = {{CALC2.0}: Combining Appearance, Semantic and Geometric Information for Robust and Efficient Visual Loop Closure},
  Author                   = {Nathaniel Merrill and Guoquan Huang},
  Booktitle                = {2019 International Conference on Intelligent Robots and Systems (IROS)},
  Year                     = {2019},
  Address                  = {Macau, China},
  Month                    = nov,
}
```
