# 1.预备工作
## 1.1 MarkDown学习

1.Markdown 语法手册 （完整整理版）：
*https://blog.csdn.net/witnessai1/article/details/52551362*

2.MarkDown公式指导手册：
*https://www.zybuluo.com/codeep/note/163962*

3.MarkDown公式编辑学习笔记：
*https://www.cnblogs.com/q735613050/p/7253073.html*

4.在线MarkDown公式编辑器：
*http://latex.codecogs.com/eqneditor/editor.php*

# 2.Anaconda

Anaconda介绍、安装及使用教程
*https://zhuanlan.zhihu.com/p/32925500*

## 2.1 Anaconda介绍

> - Anaconda是一个包含180+的科学包及其依赖项的发行版本。
> - conda是包及其依赖项和环境的管理工具；可以快速安装、运行和升级包及其依赖项，也能快捷创建、保存、加载和切换环境。conda包和环境管理器包含于Anaconda的所有版本中。
> - pip是用于安装和管理软件包的包管理器。

## 2.2 Anaconda安装
推荐使用命令行的方式安装：</br>
1. 操作系统（windows、mac；32位、64位）</br>
2. Python版本

## 2.3 管理conda
1.验证conda已被安装 </br>		
		
	conda --version
		
2.卸载conda

	rm -rf ~/anaconda2

或
	
	rm -rf ~/anaconda3
	
3.环境管理（后续补充）</br>
	
## 2.3 管理包
1.查找可供安装的包版本</br>

- 精确查找

		conda search --full-name <package_full_name>
	
	- 如 ***conda search --full-name python*** 即查找全名为“python”的包有哪些版本可供安装。

- 模糊查找
 
		conda search <text>
	
	- 如 ***conda search py*** 即查找含有“py”字段的包，有哪些版本可供安装。


2.卸载包</br>

	conda remove <package_name>
	
# 3.Jupyter Notebook

Jupyter Notebook介绍、安装及使用教程
*https://www.jianshu.com/p/91365f343585#conda*

## 3.1 安装及配置
推荐使用Anaconda的方式安装：</br>

	conda install jupyter notebook
	
## 3.2 基本使用
详见上述链接。


# 4.Tensorflow入门
TensorFlow学习笔记1：入门 </br>
*http://www.jeyzhang.com/tensorflow-learning-notes.html*

## 4.1 数据流图（Data Flow Graph）
数据流图是描述 **有向图** 中的数值计算过程。**有向图**中的节点通常代表数学运算，但也可以表示数据的输入、输出和读写等操作；**有向图**中的边表示节点之间的某种联系，它负责传输多维数据(Tensors)。

## 4.2 基本使用
你需要理解在TensorFlow中，是如何：

- 将计算流程表示成图；
- 通过 **Sessions** 来执行图计算；
- 将数据表示为 **tensors**；
- 使用 **Variables** 来保持状态信息；
- 分别使用 **feeds** 和 **fetches** 来填充数据和抓取任意的操作结果；



