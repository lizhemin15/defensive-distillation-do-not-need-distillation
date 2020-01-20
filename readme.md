# Defensive distillation do not need distillation
Since the defensive distillation was proposed, it has been widely used because of its
good defense effect against gradient attacks. This article points out that the distillation process
is actually redundant, and the reason for the failure of the gradient attack is only related to the
difference between the two maximum ligits. The increase in temperature and the learning of the
studentnetworktotheteachernetworkareredundant. However, theincreaseintemperaturecan
increase the difference in logits faster. The minimum logits difference that makes the gradient
attack invalid is derived from the theoretical derivation. Based on the proportional relationship
between logits and temperature, an estimation algorithm for quickly finding the lowest temper-
ature is designed.

# How to use
* Replace your files in C:\Users\***\Anaconda3\Lib\site-packages\keras\layers\advanced_activations.py
* Or linux:/home/lzm/miniconda3/envs/tf1.12/lib/python3.6/site-packages/keras
* You can test our result by training the models with train_mnist.py, this codes will generate some h5 files. 
* Then, run test_adv.py.
* Finally you will get some txt which record in test_acc, adv_acc, max_logits, sec_logits, max_gradient.
* You can check the result in papers by running all the codes.
