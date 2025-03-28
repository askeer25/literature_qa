# Literature QA - 学术论文问答系统

基于RAG（检索增强生成）的学术论文智能问答系统，可以帮助研究人员快速分析、检索和提取学术论文中的知识。

## 功能特点

- 支持PDF、MD和TXT格式的学术论文导入
- 自动处理和分析文档，建立向量索引
- 基于语义检索技术查询相关内容
- 智能生成论文摘要和分析
- 回答关于论文内容的特定问题
- 对比分析多篇论文的方法和结论

## 环境要求

- Python 3.8+
- OpenAI API密钥
- 必要的Python库（见requirements.txt）

## 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/askeer25/literature_qa.git
cd literature_qa
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建`.env`文件并添加以下内容：
```
OPENAI_API_KEY=你的OpenAI API密钥
OPENAI_BASE_URL=https://api.openai.com/v1 # 或自定义API服务地址
```

## 使用方法

### 基本用法

```python
import dotenv
from literature_qa import ScholarLiteratureQA

# 加载环境变量
dotenv.load_dotenv()

# 初始化系统
rag = ScholarLiteratureQA(
    path="你的论文目录",
    index_path="索引存储路径",
    llm_model_name="gpt-4o-2024-11-20",  # 或其他兼容的模型
    embedding_model_name="text-embedding-3-large"
)

# 初始化库，加载所有论文
processed_files = rag.init_library()
print(f"已处理 {len(processed_files)} 篇论文文档")

# 查询论文内容
response = rag.generate_response("你的问题", top_k=4)
print(response)

# 针对特定论文进行查询分析
paper_analysis = rag.query_paper("论文标题")
print(paper_analysis)
```

### 自定义查询

您可以调整以下参数来优化查询结果：
- `top_k`: 检索的相关文档数量
- `query_str`: 查询问题的详细描述

## 项目结构

- `literature_qa.py`: 主要代码文件
- `INDEX/`: 向量存储和索引目录
- `示例论文/`: 示例PDF论文目录

## 贡献指南

欢迎提交问题和改进建议！请通过GitHub Issues或Pull Requests参与项目。

## 许可证

MIT License