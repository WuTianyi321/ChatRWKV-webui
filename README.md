# ChatRWKV-webui

基于RWKV的webui，安装gradio、rwkv以及pytorch>=1.13之后，
`python app.py`即可启动webui。

模型从https://huggingface.co/BlinkDL
下载，具体可参考https://zhuanlan.zhihu.com/p/618011122

目前各项设置是直接写在app.py中的，模型下载好后修改args.MODEL_NAME的值为你下载的模型的路径即可。

目前generate标签页是不起作用的。
![image](https://user-images.githubusercontent.com/48122470/232069779-e84db9bb-86d1-4a10-8a3c-7f5674631f49.png)

未来计划：
1. generate完成
2. 预设prompt
3. 设置外置，文件化
3. 在webui中选择模型进行加载
