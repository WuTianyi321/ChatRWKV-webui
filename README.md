# ChatRWKV-webui

基于RWKV的webui，安装gradio、rwkv以及pytorch>=1.13之后，
`python app.py`即可启动webui。

模型从https://huggingface.co/BlinkDL
下载，具体可参考https://zhuanlan.zhihu.com/p/618011122

目前各项设置是直接写在app.py中的，模型下载好后修改args.MODEL_NAME的值为你下载的模型的路径即可。

目前generate标签页是不起作用的。
