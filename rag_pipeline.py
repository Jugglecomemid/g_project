# 思路
# 解析pdf文档，尽可能还原版面，分块
# 文本向量化，建立索引
# 使用语义模型进行精准检索并给予大模型参考回答
import re
import openai
import openparse
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

# python version 3.9
# 主要使用 openparse, llama_index 等工具构建向量索引
# 设定主要参数，例如分块的参数以及检索模型的选择（这里选择了 bge-m3）
text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=50)
Settings.text_splitter = text_splitter
Settings.embed_model = HuggingFaceEmbedding(
    model_name="model/bge-m3", device='cuda:3'
)

def _clean(text):
    '''
    简单文本清洗
    '''
    text = re.sub(r'\- \n', '- ', text)
    return text

def recover_layout(parsed_document):
    """
    版面还原代码，优先级是先从左到右，再从上到下;
    具体版面分割效果可以看 layout.ipynb
    """
    tmp_page = []
    full_text = ''
    page_use = []
    for node in parsed_document.nodes:
        if node.bbox[0].page not in page_use:
            if tmp_page:
                # 获取版面坐标值并重新排序
                sorted_a = sorted(tmp_page, key=lambda coord: (coord[0], -coord[1]))
                tmp_text = "\n".join([x[2] for x in sorted_a])
                full_text += tmp_text

            tmp_page = []
            page_use.append(node.bbox[0].page)

        tmp_page.append([node.bbox[0].x0, node.bbox[0].y1, node.text])
        # display(node.bbox)
        # display(node.text)

    with open('./txt/process_pdf.txt', 'a+') as f:
        f.write(full_text)

    return full_text

def extract_pdf(pdf_path:str):
    '''
    解析 pdf 文档
    '''
    parser = openparse.DocumentParser()
    parsed_basic_doc = parser.parse(pdf_path)
    recovered_layout_doc = recover_layout(parsed_basic_doc)
    return recovered_layout_doc

def build_index(file_director:str):
    '''
    建立索引，注意输入参数是文件夹
    '''
    documents = SimpleDirectoryReader(file_director).load_data()
    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
    # 存储向量化文件, 方便后续使用
    store_index(index)

    return index

def store_index(vector_index: VectorStoreIndex):
    '''
    向量文件持久化。默认保存在 ./storage
    '''
    vector_index.storage_context.persist()
    print('存储成功')

def load_index():
    '''
    读取在本地的向量化文件
    '''
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    vector_index = load_index_from_storage(storage_context)
    return vector_index

def recall_chunk(vector_index:VectorStoreIndex, topk:int, query:str):
    """
    召回并构造用于传入大模型的prompt
    """
    chunk_prompt = ''
    base_retriever = vector_index.as_retriever(similarity_top_k=topk)
    results = base_retriever.retrieve(query)
    ref_i = 1
    for result in results:
        chunk_prompt = chunk_prompt + f'[信息{ref_i}]' + result.text + '\n'
        ref_i += 1
    print(chunk_prompt)
    return chunk_prompt

def request_llm(message_log:list):
    """
    大模型使用 fastchat 框架部署;
    具体部署流程请参考 README.md;
    好处是兼容 openai 调用方式，易于管理多个大模型实例部署;
    """
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:8000/v1/"

    model = "Qwen2-0.5B-Instruct"
    prompt = "Once upon a time"
    for chunk in openai.chat.completions.create(
        model=model,
        messages=message_log,
        temperature=0.7,
        max_tokens=2048,
        stream=True
        ):
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True) 

def main():
    pdf_path = 'pdf/pp_yolo.pdf'
    txt_path = 'txt'
    test_query = 'What tricks are used when training pp-yole?'
    update_index = True

    processed_pdf = extract_pdf(pdf_path)
    if update_index:
        vector_index = build_index(txt_path)
    else:
        vector_index = load_index()
    # 召回4个检索片段
    ref_prompt = recall_chunk(vector_index, 4, test_query)

    prompt_tempate = f'你要参考信息回答问题。参考信息：{ref_prompt}。 要求：只能使用参考信息中的内容回答，不可以编造答案。\n 问题：{test_query}'
    message_log = [
        {'role':'system', 'content':'你是一个问答助手。你能参考信息回答用户的问题。你不会编造答案'},
        {'role':'user', 'content':prompt_tempate}
    ]
    request_llm(message_log)

if __name__ == '__main__':
    main()
    # model answer: We believe that using a better backbone network, using more effective data augmentation method and using NAS to search for hyperparameters can further improve the per-formance of PP-YOLO.
