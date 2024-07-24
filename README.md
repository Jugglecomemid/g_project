# 项目结构
.  
├── layout.ipynb （版面分割效果）  
├── booms.py  （题目一代码）  
├── model  
│   ├── bge-m3  
│   └── Qwen2-0.5B-Instruct  
├── pdf  
│   └── pp_yolo.pdf  
├── rag_pipeline.py  (题目二代码)  
├── README.md  
├── requirments.txt  
├── resume  （简历）  
├── storage  
└── txt  （还原版面后的pdf）


# 项目一
### 项目简述
详见 booms.py

# 项目二
### 项目简述
1. 代码及注释详见 rag_pipeline.py。
2. RAG检索工具主要使用 **openparse** 协助解析 pdf 和还原 pdf 版面，使用 **llama_index** 来进行文档分割和构建向量索引。
3. 大模型部署框架采用 **fastchat**。部署详细步骤可看下面的 **模型部署** 部分。
4. 使用了 **Qwen2-0.5B-Instruct** 而非纯粹的 **Qwen2-0.5B**，主要原因是后者是一个预训练的续写模型，未经过任何指令微调，时间关系改用了经过指令微调的前者。
5. 语义匹配模型选用了 **bge-m3**，支持多语言的语义匹配，目前在向量匹配任务上的排名非常高。
6. 由于模型文件较大，没有上传到 github 上。所有模型均保存在 **./model** 路径下。详见 **模型下载**。


### 模型下载
```
cd model
git lfs install

# 能直连 huggingface
git clone https://huggingface.co/BAAI/bge-m3
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct

# 国内镜像（速度快）
git clone https://www.modelscope.cn/Xorbits/bge-m3.git
git clone https://www.modelscope.cn/qwen/Qwen2-0.5B-Instruct.git

```

### 模型部署
```
pip install -r requirement.txt

# 启动 Controller, 功能是注册 Worker 以及为请求随机分配 Worker
python3 -m fastchat.serve.controller

# 启动 Worker, 提供模型api接口
python3 -m fastchat.serve.model_worker --model-path model/Qwen2-0.5B-Instruct

# 启动 openai 接口，允许以 openai 方式调用所有模型
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000

```

### 优化方向
1. 文档分割效果待验证。理论上使用标点符号或者换行符分割更能保证文档完整度。
2. 文档检索出来后缺少排序阶段。实际应该召回更多批次，在批次排序并选择topk。
3. 文档构建方式可改用树结构查询。因为此类论文具有比较强的 标题-正文 属性，可通过标题，副标题甚至总结正文的关系构建树结构进行精确查询。离线下，可借助大模型进行理解总结。
3. 问答大模型可选用更大尺寸如7B或14B，理解能力更强，延时也能够接受。
