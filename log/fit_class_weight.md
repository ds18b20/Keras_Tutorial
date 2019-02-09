# Keras fit

parameter: class_weight

> class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）

目的：
用于在不平衡数据上的训练时合理调整各类别的更新速度。

最简单的应用方式为：
将该参数设为字符串class_weight = 'auto'。

设置了这个参数后，keras会自动设置class weight让每类的sample对损失的贡献相等。
https://www.jianshu.com/p/4d2c648bc589

另外也可以使用字典手动配置：

So if you have 3 classes with classA:10%, classB:50% and classC:40% then you get the weights:

```
{0:5, 1:1, 2:1.25}
```
https://github.com/keras-team/keras/issues/1875#issuecomment-273752868
https://stackoverflow.com/questions/44716150/how-can-i-assign-a-class-weight-in-keras-in-a-simple-way/44721883#44721883


