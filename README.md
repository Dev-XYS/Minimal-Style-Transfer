# 水墨画图像风格迁移

代码由 [https://github.com/yunjey/mnist-svhn-transfer](https://github.com/yunjey/mnist-svhn-transfer) 修改而来。

## 训练

将训练集中的照片放入 `data/horse/trainA`，水墨画放入 `data/horse/trainB`，然后运行
``` bash
python3 src/main.py
```
每 5000 次迭代会保存模型至 `output/models`。

## 生成

使用如下命令
``` bash
python3 src/main.py --mode sample --model_path <model-path> --photo_path <photo-path>
```
请将上述命令中 `<model-path>` 替换为模型路径，将 `<photo-path>` 替换为输入图片所在目录，程序会将此目录下的所有图片进行风格迁移，并保存至 `output/samples`。

## 移动端模型生成

使用如下命令
``` bash
python3 src/main.py --mode gen_mobile_model --model_path <model-path>
```
输出为当前目录下的 `g21.pt`。
