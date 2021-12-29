
# YOLOv3_Detect_Web
Use Yolov3 detect on Web

![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229095207661-1020206979.png)

使用 YOLOv3（PyTorch 和 Django 实现）的对象检测应用程序。
网页和 REST API由Django Web框架实现。

# 1. Introduction 介绍

---

这是一个使用 YOLOv3 提供对象检测并生成 REST API 的 Web 应用程序。
它是使用 Django 框架和 PyTorch（用于 YOLO 模型）实现的。
这里开发了接受图像作为请求的 Django API，API 的响应是 JSON 对象。
输入图像被转换为 float32 类型的 NumPy 数组并传递给 YOLOv3 对象检测模型。
该模型对图像执行对象检测，并生成一个 JSON 对象，其中包含所有对象的名称及其在图像中各自的置信度。


# 2. Required Libraries 依赖库对应版本及环境配置

---

## 2.1  所需依赖库

下面提到了所需的库及其版本：

* Python  (3.7)
* Django  (3.0.3)
* PyTorch (1.3.1)
* Pillow  (7.1.2)
* OpenCV  (4.2.0)
* NumPy   (1.18.5)

可见requirements.txt。

## 2.2 配置测试环境
- 利用Anaconda创建名为web的虚拟环境

```shell
conda create -n web python=3.7
```
- 进入虚拟环境

```shell
conda activate web
```
- 根据requirements文件在清华源下进行依赖库安装(推荐使用)

```shell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 3. Required files for Detection 检测必需的文件

---
要使用预训练模型的对象检测，我们需要三个重要文件，分别为以下：

- **yolov3.cfg - cfg：**
  该文件用来逐块描述网络的布局。官方 cfg 文件可在Darknet github 存储库中找到。
  但是，为了获得更好的性能，我对配置文件做了一些更改。
- **yolov3.weights:**
  我们使用来自 darknet53 模型的权重。
- **coco.names:**
  文件包含我们的模型经过训练可以识别的不同对象的名称。

# 4.Steps to Follow (Working)

---

这个存储库可以做两件事：

1. 基于YOLOv3和Django的网页程序应用实现
2. REST API的生成（API测试使用POSTMAN完成）

## 4.1 网页应用实现

- **step-1.**克隆 GitHub 存储库

```shell
git clone https://github.com/isLinXu/YOLOv3_Detect_Web.git
```

- **step-2.**将目录更改为克隆的 Repository 文件夹。

```shell
cd YOLOv3_Detect_Web
```

- **step-3**.由于.cfg 和 coco.names 文件已在此存储库中默认设置好， 可根据需要进行自行修改。
  现在，我们需要做的就是下载权重文件。
  在命令提示符中使用以下命令下载 yolov3.weights：

```shell
wget https://pjreddie.com/media/files/yolov3.weights
```

- **step-4**.安装所有必需的库。
- **step-5.**执行下面的代码：（这条命令只需要执行一次，用来初始化创建）

```shell
python manage.py collectstatic
```

此命令启动 Django 并收集所有静态文件。

- **step-6**.然后，开始服务：

```shell
python manage.py runserver
```

此命令启动 Django 服务器。

现在我们都准备好运行应用程序了。

- **step-7**..执行上述代码后，您将看到如下内容：

![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229102043283-1298501310.png)

- **step-8.**点击链接。这会将您定向到 Web 浏览器。

  ![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229102544422-65710418.png)

- **step-9**.通过拖放或浏览模式选择图像。

  ![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229102538414-994383395.png)

- **setp-10**:上传图片

![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229111240890-1808341010.png)
![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229112155005-1239105137.png)


- **step-11**: 点击DEDECT-OBJECT，进行检测图片，这时会将结果解析为json并显示出来。
  Django Web-app 的输入是一个图像。此输入图像被转换为​​ float32 类型的 NumPy 数组并传递给 YOLOv3 模型。
  该模型对图像执行对象检测，并生成一个 JSON 对象，其中包含所有对象的名称及其在图像中各自的频率。
  
  ![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229111348163-1486954962.png)
  表单响应是 JSON 对象。此 JSON 对象如上所示显示。

- **step-12:** 单击“Show Predictions”显示检测结果,查看带有边界框的图像。

![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229111605379-1714549773.png)

- **step-13:** 要尝试其他图像，请单击"Choose a New File"

## 4.2 REST API 实现——POSTMAN

Postman 是一个可扩展的 API 测试工具。要遵循的步骤是：

1. 按照上面提到的前 6 个步骤进行操作。

2. 确保服务器正常运行

   ![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229112728941-539819044.png)

3. 打开 POSTMAN 并选择 POST 选项。输入上面显示的服务器链接并将 /object_detection/api_request/ 附加到它。

4. 点击body,输入key value作为"image"，选择图片文件点击“Send”进行发送

5. 输入图像被转换为 float32 类型的 NumPy 数组并传递给 YOLOv3 模型。该模型对图像执行对象检测，并生成一个 JSON 对象，其中包含所有对象的名称及其在图像中各自的频率。

   ![](https://img2020.cnblogs.com/blog/1571518/202112/1571518-20211229115830026-325581645.png)

6. HttpResponse 是 JSON 对象。其中此 JSON 对象如上所示显示。

例如：127.0.0.1:8000/object_detection/api_request/





