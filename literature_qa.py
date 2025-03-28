import os
import logging
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import TextNode, Document, NodeWithScore
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import PromptTemplate
from typing import Optional, Any, List, Dict, Set, Tuple, Union
from pymupdf import EmptyFileError
import chromadb

logger = logging.getLogger("__name__")

class DocumentLoader:
    """负责加载和解析不同格式的文档"""
    
    def __init__(self):
        self.pdf_loader = PyMuPDFReader()
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档并返回Document对象列表"""
        try:
            if file_path.lower().endswith(".pdf"):
                documents = self.pdf_loader.load(file_path=file_path)
            elif file_path.lower().endswith((".md", ".txt")):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                metadata = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": "md" if file_path.lower().endswith(".md") else "txt",
                }
                documents = [Document(text=text, metadata=metadata)]
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                return []

            return documents
        except EmptyFileError as e:
            logger.error(f"无法打开空文件: {file_path}")
            logger.warning(f"移除空文件: {file_path}")
            os.remove(file_path)
            return []
        except Exception as e:
            logger.error(f"加载文件 {file_path} 失败: {str(e)}")
            os.remove(file_path)
            logger.warning(f"移除有问题的文件: {file_path}")
            return []


class DocumentProcessor:
    """负责文档的切分和处理"""
    
    def __init__(self, chunk_size: int = 1024):
        self.text_splitter = TokenTextSplitter(chunk_size=chunk_size)
    
    def process_documents(self, documents: List[Document]) -> Tuple[List[str], List[int]]:
        """将文档切分成块并返回文本块和对应的文档索引"""
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = self.text_splitter.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        return text_chunks, doc_idxs
    
    def create_nodes(self, documents: List[Document], doc_idxs: List[int], 
                    text_chunks: List[str]) -> List[TextNode]:
        """根据文本块创建TextNode对象"""
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
        return nodes


class EmbeddingService:
    """负责文本向量化"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, 
                api_base: Optional[str] = None, dimensions: int = 512):
        self.model = OpenAIEmbedding(
            model=model_name,
            dimensions=dimensions,
            api_key=api_key,
            api_base=api_base,
        )
    
    def embed_nodes(self, nodes: List[TextNode], batch_size: int = 50) -> List[TextNode]:
        """对节点进行向量化"""
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            try:
                for node in batch:
                    node_embedding = self.model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding
                logger.info(
                    f"处理嵌入向量批次 {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}"
                )
            except Exception as e:
                logger.error(
                    f"为批次 {i//batch_size + 1} 创建嵌入向量时出错: {e}"
                )
        return nodes
    
    def get_query_embedding(self, query: str) -> List[float]:
        """获取查询的嵌入向量"""
        return self.model.get_query_embedding(query)


class VectorStore:
    """负责向量数据库的管理"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)
        self.vector_store = self._init_vector_store()
        self.processed_files = self._load_processed_files()
    
    def _init_vector_store(self):
        """初始化向量存储"""
        chroma_client = chromadb.PersistentClient(path=self.index_path)
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store
    
    def _load_processed_files(self) -> Set[str]:
        """加载已处理过的文件列表"""
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r") as f:
                files = [line.strip() for line in f.readlines()]
            return set(files)
        else:
            return set()
    
    def save_processed_files(self):
        """保存已处理过的文件列表"""
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        with open(processed_file_path, "w") as f:
            for file in self.processed_files:
                f.write(file + "\n")
    
    def add_nodes(self, nodes: List[TextNode]):
        """添加节点到向量存储"""
        self.vector_store.add(nodes)
    
    def mark_as_processed(self, file_path: str):
        """标记文件为已处理"""
        self.processed_files.add(file_path)
    
    def is_processed(self, file_path: str) -> bool:
        """检查文件是否已处理"""
        return file_path in self.processed_files
    
    def get_vector_store(self):
        """获取原始向量存储对象"""
        return self.vector_store


class VectorDBRetriever(BaseRetriever):
    """负责从向量数据库中检索相关内容"""
    
    def __init__(
        self,
        vector_store: Any,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_str: str) -> List[NodeWithScore]:
        """检索相关内容"""
        query_embedding = self._embed_model.get_query_embedding(query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores


class ResponseGenerator:
    """负责生成最终回答"""
    
    def __init__(self, llm_model_name: str, api_key: Optional[str] = None, 
                api_base: Optional[str] = None):
        self.llm = OpenAI(
            model=llm_model_name,
            api_key=api_key,
            api_base=api_base,
            max_tokens=4096,
            temperature=0.4,
        )
        self.academic_paper_prompt = PromptTemplate(
            """\
You are an expert research assistant with deep knowledge in scientific literature analysis.

QUERY: {query_str}

CONTEXT INFORMATION:
------------
{context_str}
------------

Based on the provided academic context, please respond to the query with precise information. When analyzing papers:
1. Identify key contributions, methodologies, and findings
2. Explain technical concepts clearly and accurately
3. Highlight relationships to existing research if mentioned
4. Note limitations or future work discussed
5. Organize your response in a logical structure

NOTE: 使用中文回答！

If the context doesn't contain sufficient information to answer the query, clearly state what information is missing.
If information appears contradictory or unclear, acknowledge this and provide the most reasonable interpretation.

Your response should be comprehensive, academically rigorous, yet accessible.
"""
        )
    
    def generate_response(self, query_str: str, context_nodes: List[NodeWithScore]) -> str:
        """根据查询和上下文生成回答"""
        if not context_nodes:
            logger.warning("未找到与查询相关的节点")
            return "未找到相关信息。"

        context_str = "\n\n".join([r.node.get_content() for r in context_nodes])
        
        fmt_prompt = self.academic_paper_prompt.format(
            context_str=context_str,
            query_str=query_str,
        )

        try:
            response = self.llm.complete(fmt_prompt)
            logger.info(f"为查询生成回答: {query_str}")
            return str(response)
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            return "生成回答时发生错误。"


class ScholarRAG:
    """主类，组合并管理所有RAG组件"""
    
    def __init__(
        self,
        path: str,
        index_path: str,
        llm_model_name: str,
        embedding_model_name: str,
    ) -> None:
        # 初始化路径
        self.path = path
        self.index_path = index_path
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # 获取API密钥
        llm_api_key = os.getenv("OPENAI_API_KEY")
        llm_api_base = os.getenv("OPENAI_BASE_URL")
        embed_api_key = os.getenv("OPENAI_API_KEY")
        embed_api_base = os.getenv("OPENAI_BASE_URL")
        
        # 初始化各组件
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor(chunk_size=1024)
        self.embedding_service = EmbeddingService(
            model_name=embedding_model_name,
            api_key=embed_api_key,
            api_base=embed_api_base,
        )
        self.vector_store = VectorStore(index_path=self.index_path)
        self.response_generator = ResponseGenerator(
            llm_model_name=llm_model_name,
            api_key=llm_api_key,
            api_base=llm_api_base,
        )
    
    def process_file(self, file_path: str) -> bool:
        """处理单个文件"""
        if self.vector_store.is_processed(file_path):
            logger.info(f"文件已处理过，跳过: {file_path}")
            return False
            
        try:
            # 加载文档
            documents = self.document_loader.load_document(file_path)
            if not documents:
                return False
                
            # 处理文档
            text_chunks, doc_idxs = self.document_processor.process_documents(documents)
            nodes = self.document_processor.create_nodes(documents, doc_idxs, text_chunks)
            
            # 向量化
            nodes = self.embedding_service.embed_nodes(nodes)
            
            # 存储向量
            self.vector_store.add_nodes(nodes)
            self.vector_store.mark_as_processed(file_path)
            logger.info(f"已处理文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return False
    
    def init_library(self) -> List[str]:
        """初始化文档库，处理所有文件"""
        supported_extensions = [".pdf", ".md", ".txt"]
        file_list = []
        for filepath, _, filenames in os.walk(self.path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(filepath, filename)
                    if not self.vector_store.is_processed(file_path):
                        file_list.append(file_path)
        
        processed_count = 0
        for file_path in file_list:
            if self.process_file(file_path):
                processed_count += 1
        
        self.vector_store.save_processed_files()
        logger.info(f"初始化时处理了 {processed_count} 个文件")
        return file_list
    
    def retrieve(self, query_str: str, top_k: int = 5) -> List[NodeWithScore]:
        """检索相关内容"""
        retriever = VectorDBRetriever(
            self.vector_store.get_vector_store(),
            self.embedding_service,
            query_mode="default",
            similarity_top_k=top_k,
        )
        retrieved_nodes = retriever._retrieve(query_str)
        logger.info(f"为查询检索到 {len(retrieved_nodes)} 个节点")
        return retrieved_nodes
    
    def generate_response(self, query_str: str, top_k: int = 5) -> str:
        """生成回答"""
        # 检索阶段
        retrieved_nodes = self.retrieve(query_str, top_k)
        
        # 生成阶段
        response = self.response_generator.generate_response(query_str, retrieved_nodes)
        return response
    
    def query_paper(self, query: str) -> str:
        """根据用户查询针对特定论文进行检索和回答生成
        
        如果用户查询意图是针对特定论文，则只检索该论文的内容；
        否则进行正常的向量检索
        """
        # 判断是否是查询特定论文的意图
        import re
        
        # 识别查询中是否包含明确的论文标题
        paper_title_patterns = [
            r'《(.+?)》',  # 中文引号
            r'"(.+?)"',    # 英文双引号
            r"'(.+?)'",    # 英文单引号 (修复：使用标准直引号替代弯引号)
        ]
        
        paper_title = None
        for pattern in paper_title_patterns:
            match = re.search(pattern, query)
            if match:
                paper_title = match.group(1)
                logger.info(f"从查询中识别出论文标题: {paper_title}")
                break
        
        # 如果没有从引号中识别出标题，尝试从关键词模式中识别
        if not paper_title:
            paper_keywords = ["论文", "paper", "文章", "article", "publication"]
            has_paper_keyword = any(keyword in query for keyword in paper_keywords)
            
            if has_paper_keyword:
                # 从系统中获取所有已处理的论文文件路径
                paper_file_paths = list(self.vector_store.processed_files)
                
                # 从文件路径中提取论文标题
                paper_titles = []
                for file_path in paper_file_paths:
                    if os.path.isfile(file_path) and file_path.lower().endswith((".pdf", ".md", ".txt")):
                        file_name = os.path.basename(file_path)
                        # 移除扩展名和可能的前缀路径
                        title = os.path.splitext(file_name)[0]
                        paper_titles.append((title, file_path))
                
                # 计算查询与每个论文标题的相似度
                if paper_titles:
                    best_match = None
                    best_score = -1
                    
                    # 使用嵌入模型计算相似度
                    query_embedding = self.embedding_service.get_query_embedding(query)
                    
                    for title, file_path in paper_titles:
                        title_embedding = self.embedding_service.get_query_embedding(title)
                        
                        # 计算余弦相似度
                        import numpy as np
                        similarity = np.dot(query_embedding, title_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(title_embedding)
                        )
                        
                        if similarity > best_score:
                            best_score = similarity
                            best_match = (title, file_path)
                    
                    # 如果最佳匹配相似度超过阈值，认为用户意图是查询该论文
                    if best_match and best_score > 0.6:
                        paper_title = best_match[0]
                        logger.info(f"从查询中匹配到最相关论文: {paper_title}, 相似度: {best_score}")
        
        if paper_title:
            # 针对特定论文进行检索
            logger.info(f"针对特定论文进行查询: {paper_title}")
            
            # 构建过滤器函数来只检索特定论文的内容
            def filter_nodes_by_paper(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
                filtered_nodes = []
                for node in nodes:
                    # 检查节点元数据中的文件名或路径是否包含论文标题
                    metadata = node.node.metadata
                    if "file_name" in metadata and paper_title.lower() in metadata["file_name"].lower():
                        filtered_nodes.append(node)
                    elif "file_path" in metadata and paper_title.lower() in metadata["file_path"].lower():
                        filtered_nodes.append(node)
                return filtered_nodes
            
            # 先进行较大范围的检索，然后过滤
            retrieved_nodes = self.retrieve(query, top_k=20)
            filtered_nodes = filter_nodes_by_paper(retrieved_nodes)
            
            # 如果过滤后没有结果，使用更宽泛的检索策略
            if not filtered_nodes:
                logger.warning(f"未找到与论文《{paper_title}》相关的内容，尝试更宽泛的检索")
                paper_query = f"关于论文《{paper_title}》的内容"
                retrieved_nodes = self.retrieve(paper_query, top_k=15)
                filtered_nodes = filter_nodes_by_paper(retrieved_nodes)
            
            # 生成特定论文的回答
            if filtered_nodes:
                logger.info(f"为论文《{paper_title}》找到 {len(filtered_nodes)} 个相关节点")
                response = self.response_generator.generate_response(query, filtered_nodes)
                return response
            else:
                return f"很抱歉，未能在知识库中找到与论文《{paper_title}》相关的内容。"
        
        # 如果不是查询特定论文的意图，进行常规检索和回答生成
        logger.info("进行常规查询检索")
        return self.generate_response(query, top_k=8)


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    llm_model_name = "gpt-4o-2024-11-20"
    embedding_model_name = "text-embedding-3-large"

    rag = ScholarRAG(
        path="示例论文",
        index_path="INDEX",
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
    )
    
    # 初始化库，加载所有论文
    processed_files = rag.init_library()
    print(f"已处理 {len(processed_files)} 篇论文文档")

    # 测试用例1: 查询AlphaFold的基本原理
    print("\n--- 测试用例1: AlphaFold基本原理 ---")
    response1 = rag.generate_response("解释AlphaFold的基本原理和创新点是什么?", top_k=4)
    print(response1)

    # 测试用例2: 针对特定论文进行查询分析
    print("\n--- 测试用例2: 特定论文分析 ---")
    response2 = rag.query_paper("\"Highly accurate protein structure prediction with AlphaFold\"这篇论文的核心创新点是什么？")
    print(response2)


