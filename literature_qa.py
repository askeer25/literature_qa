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
from typing import Optional, Any, List
from pymupdf import EmptyFileError
import chromadb

logger = logging.getLogger("__name__")

class VectorDBRetriever(BaseRetriever):
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

class ScholarLiteratureQA:
    def __init__(
        self,
        path: str,
        index_path: str,
        llm_model_name: str,
        embedding_model_name: str,
    ) -> None:
        self.path = path
        self.index_path = index_path
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        llm_api_key = os.getenv("OPENAI_API_KEY")
        llm_api_base = os.getenv("OPENAI_BASE_URL")
        self.llm_model = OpenAI(
            model=llm_model_name,
            api_key=llm_api_key,
            api_base=llm_api_base,
            max_tokens=4096,
            temperature=0.4,
        )
        embed_api_key = os.getenv("OPENAI_API_KEY")
        embed_api_base = os.getenv("OPENAI_BASE_URL")

        self.embedding_model = OpenAIEmbedding(
            model=embedding_model_name,
            dimensions=512,
            api_key=embed_api_key,
            api_base=embed_api_base,
        )

        if self.llm_model is None or self.embedding_model is None:
            raise ValueError("Both LLM and embedding model are not initialized.")

        self.vector_store = self.init_vector_store()
        self.processed_files = self.load_processed_files()

    def init_vector_store(self):
        chroma_client = chromadb.PersistentClient(path=self.index_path)
        chroma_collection = chroma_client.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        return vector_store

    def load_processed_files(self):
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        if os.path.exists(processed_file_path):
            with open(processed_file_path, "r") as f:
                files = [line.strip() for line in f.readlines()]
            return set(files)
        else:
            return set()

    def save_processed_files(self):
        processed_file_path = os.path.join(self.index_path, "processed_files.txt")
        with open(processed_file_path, "w") as f:
            for file in self.processed_files:
                f.write(file + "\n")

    def load_data(self, path: str) -> List[Document]:
        try:
            if path.lower().endswith(".pdf"):
                loader = PyMuPDFReader()
                documents = loader.load(file_path=path)
            elif path.lower().endswith((".md", ".txt")):
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                metadata = {
                    "file_path": path,
                    "file_name": os.path.basename(path),
                    "file_type": "md" if path.lower().endswith(".md") else "txt",
                }
                documents = [Document(text=text, metadata=metadata)]

            else:
                logger.warning(f"Unsupported file format: {path}")
                return []

            return documents
        except EmptyFileError as e:
            logger.error(f"Cannot open empty file: {path}")
            logger.warning(f"Removing empty file: {path}")
            os.remove(path)
            return []
        except Exception as e:
            logger.error(f"Failed to load {path}: {str(e)}")
            os.remove(path)
            logger.warning(f"Removing problematic file: {path}")
            return []

    def chunk_documents(self, documents: List[Document]) -> List[int]:
        text_parser = TokenTextSplitter(chunk_size=1024)
        text_chunks = []
        doc_idxs = []
        for doc_idx, doc in enumerate(documents):
            cur_text_chunks = text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)
            doc_idxs.extend([doc_idx] * len(cur_text_chunks))
        return text_chunks, doc_idxs

    def create_nodes(
        self, documents: List[Document], doc_idxs: List[int], text_chunks: List[str]
    ) -> List[TextNode]:
        nodes = []
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(text=text_chunk)
            src_doc = documents[doc_idxs[idx]]
            node.metadata = src_doc.metadata
            nodes.append(node)
        return nodes

    def create_embeddings(self, nodes: List[TextNode]) -> List[TextNode]:
        batch_size = 50
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            try:
                for node in batch:
                    node_embedding = self.embedding_model.get_text_embedding(
                        node.get_content(metadata_mode="all")
                    )
                    node.embedding = node_embedding
                logger.info(
                    f"Processed embeddings batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}"
                )
            except Exception as e:
                logger.error(
                    f"Error creating embeddings for batch {i//batch_size + 1}: {e}"
                )
        return nodes

    def store_vector(self, path: str) -> None:
        try:
            documents = self.load_data(path)
            if documents:
                text_chunks, doc_idxs = self.chunk_documents(documents)
                nodes = self.create_nodes(documents, doc_idxs, text_chunks)
                nodes = self.create_embeddings(nodes)
                self.vector_store.add(nodes)
                logger.info(f"Stored vectors for {path}")
                self.processed_files.add(path)
        except Exception as e:
            logger.error(f"Error processing file {path}: {e}")

    def process_files(self, files_to_process):
        processed_count = 0
        for file_path in files_to_process:
            try:
                self.store_vector(file_path)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        self.save_processed_files()
        return processed_count

    def init_library(self) -> List[str]:
        supported_extensions = [".pdf", ".md", ".txt"]
        file_list = []
        for filepath, _, filenames in os.walk(self.path):
            for filename in filenames:
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(filepath, filename)
                    if file_path not in self.processed_files:
                        file_list.append(file_path)
        if file_list:
            processed_count = self.process_files(file_list)
            logger.info(f"Processed {processed_count} files during initialization")
        else:
            logger.info("No files found to process during initialization")
        return file_list

    def get_retrieve(self, query_str: str, top_k: int) -> List[NodeWithScore]:
        retriever = VectorDBRetriever(
            self.vector_store,
            self.embedding_model,
            query_mode="default",
            similarity_top_k=top_k,
        )
        retrieved_nodes = retriever._retrieve(query_str)
        logger.info(f"Retrieved {len(retrieved_nodes)} nodes for query")
        return retrieved_nodes

    def generate_response(self, query_str: str = None, top_k: int = 5) -> str:
        retrieved_nodes = self.get_retrieve(query_str, top_k)
        if not retrieved_nodes:
            logger.warning("No nodes retrieved for the query.")
            return "No relevant information found."

        context_str = "\n\n".join([r.node.get_content() for r in retrieved_nodes])

        academic_paper_prompt = PromptTemplate(
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

If the context doesn't contain sufficient information to answer the query, clearly state what information is missing.
If information appears contradictory or unclear, acknowledge this and provide the most reasonable interpretation.

Your response should be comprehensive, academically rigorous, yet accessible.
"""
        )

        fmt_prompt = academic_paper_prompt.format(
            context_str=context_str,
            query_str=query_str,
        )

        try:
            response = self.llm_model.complete(fmt_prompt)
            logger.info(f"Generated response for query: {query_str}")
            return str(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while generating the response."

    def query_paper(self, paper_title: str) -> str:
        query = f"Provide a comprehensive summary and analysis of the paper titled: {paper_title}"
        logger.info(f"Querying specific paper: {paper_title}")
        return self.generate_response(query, top_k=8)

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    llm_model_name = "gpt-4o-2024-11-20"
    embedding_model_name = "text-embedding-3-large"

    rag = ScholarLiteratureQA(
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
    response2 = rag.query_paper("Highly accurate protein structure prediction with AlphaFold")
    print(response2)

    # 测试用例3: 比较不同方法的优缺点
    print("\n--- 测试用例3: 方法比较 ---")
    response3 = rag.generate_response("比较AlphaFold和HelixFold在蛋白质结构预测方面的性能差异", top_k=6)
    print(response3)


