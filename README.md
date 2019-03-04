## CALC2.0

Convolutional Autoencoder for Loop Closure 2.0.

To get started, download the COCO dataset and the "stuff" annotations, then run dataset/gen_tfrecords.py.
Make sure to unzip the tar in the dataset directory first.
Doing this will generate the sharded tfrecord files as well as loss_weights.txt.

After that you can train with calc2.py.

Check the --mode options in calc2.py to see what else you can do, like PR curves and finding the best model in a directory.
If you want to see a visualization of the descriptors run test_net.py with a trained model.
You will need the newest [scikit-cuda](https://github.com/lebedov/scikit-cuda) for this.
