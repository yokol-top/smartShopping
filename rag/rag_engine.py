import logging
import os
from typing import List, Dict, Any

from observability import get_tracer
from utils.llm_client import LLMClient
from .document_loader import DocumentLoader
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingModel
from .hybrid_retriever import HybridRetriever
from .hyde import HyDE
from .query_rewrite_router import QueryRewriteRouter
from .query_rewriter import QueryRewriter, MultiQueryGenerator, CoRefResolver
from .reranker import Reranker
from .self_fix import SelfFix
from .vector_store import VectorStore


class RAGEngine:
    """RAG引擎：整合所有RAG组件"""
    
    def __init__(self, config: Dict[str, Any], llm_client: LLMClient = None, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("初始化RAG引擎")
        
        # 初始化LLM客户端（如果未提供）
        if llm_client is None:
            llm_client = LLMClient(
                api_key=config.get('llm', {}).get('api_key', 'EMPTY'),
                base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434/v1'),
                model=config.get('llm', {}).get('model', 'llama3.2'),
                temperature=config.get('llm', {}).get('temperature', 0.7),
                max_tokens=config.get('llm', {}).get('max_tokens', 2000),
                logger=self.logger
            )
        self.llm_client = llm_client
        
        # 初始化嵌入模型
        self.embeddings = EmbeddingModel(
            provider=config.get('embedding', {}).get('provider', 'ollama'),
            model=config.get('embedding', {}).get('model', 'nomic-embed-text'),
            api_key=config.get('embedding', {}).get('api_key', 'EMPTY'),
            base_url=config.get('embedding', {}).get('base_url', None),
            logger=self.logger
        )
        
        # 初始化文档加载器
        self.doc_loader = DocumentLoader(logger=self.logger)
        
        self.vector_store = VectorStore(
            persist_directory=config.get('vectordb', {}).get('persist_directory', './data/chroma_db'),
            collection_name=config.get('vectordb', {}).get('collection_name', 'knowledge_base'),
            logger=self.logger
        )
        
        self.doc_processor = DocumentProcessor(
            parent_chunk_size=config.get('rag', {}).get('parent_child_chunking', {}).get('parent_chunk_size', 1000),
            child_chunk_size=config.get('rag', {}).get('parent_child_chunking', {}).get('child_chunk_size', 200),
            chunk_overlap=config.get('rag', {}).get('parent_child_chunking', {}).get('chunk_overlap', 50),
            logger=self.logger
        )
        
        self.rag_config = config.get('rag', {})

        # 使用统一的LLM客户端初始化所有组件
        self.query_rewriter = QueryRewriter(llm_client=self.llm_client, logger=self.logger)
        self.multi_query_gen = MultiQueryGenerator(llm_client=self.llm_client, logger=self.logger)
        self.hyde = HyDE(llm_client=self.llm_client, logger=self.logger)
        self.reranker = Reranker(llm_client=self.llm_client, logger=self.logger)

        # 查询重写路由器（4 级分流，替代原无条件重写）
        self.coref_resolver = CoRefResolver(llm_client=self.llm_client, logger=self.logger)
        self.rewrite_router = QueryRewriteRouter(
            coref_resolver=self.coref_resolver,
            semantic_rewriter=self.query_rewriter,
            hyde=self.hyde,
            multi_query_gen=self.multi_query_gen,
            config=self.rag_config,
            logger=self.logger,
        )
        self.self_fix = SelfFix(
            llm_client=self.llm_client,
            max_iterations=self.rag_config.get('self_fix', {}).get('max_iterations', 2),
            logger=self.logger
        )

        self.hybrid_retriever = HybridRetriever(
            vector_weight=self.rag_config.get('hybrid_search', {}).get('vector_weight', 0.7),
            bm25_weight=self.rag_config.get('hybrid_search', {}).get('bm25_weight', 0.3),
            logger=self.logger
        )
        
        self.logger.info("RAG引擎初始化完成")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            metadatas: 元数据列表
        """
        self.logger.info(f"添加 {len(documents)} 个文档到知识库")
        
        # 处理文档（父子分块）
        child_chunks, child_metadatas, child_ids, parent_child_map = \
            self.doc_processor.process_documents(documents, metadatas)
        
        # 生成嵌入向量
        embeddings = self.embeddings.embed_documents(child_chunks)
        
        # 添加到向量数据库
        self.vector_store.add_documents(
            documents=child_chunks,
            metadatas=child_metadatas,
            ids=child_ids,
            embeddings=embeddings
        )
        
        # 构建BM25索引
        self.hybrid_retriever.build_bm25_index(
            documents=child_chunks,
            document_ids=child_ids,
            metadatas=child_metadatas
        )
        
        self.logger.info("文档添加完成")
    
    def add_documents_from_file(self, file_path: str, metadata: Dict[str, Any] = None):
        """
        从单个文件添加文档到知识库
        
        Args:
            file_path: 文件路径
            metadata: 额外的元数据
        """
        self.logger.info(f"从文件加载文档: {file_path}")
        
        # 加载文档
        content = self.doc_loader.load_single_file(file_path)
        
        # 合并元数据
        file_metadata = {
            'source': os.path.basename(file_path),
            'file_path': file_path,
            'file_type': os.path.splitext(file_path)[1]
        }
        if metadata:
            file_metadata.update(metadata)
        
        # 添加到知识库
        self.add_documents([content], [file_metadata])
    
    def add_documents_from_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_extensions: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        从目录批量添加文档到知识库
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归处理子目录
            file_extensions: 指定文件扩展名列表
            metadata: 额外的元数据
        """
        self.logger.info(f"从目录加载文档: {directory_path}")
        
        # 加载目录中的所有文档
        docs = self.doc_loader.load_directory(
            directory_path,
            recursive=recursive,
            file_extensions=file_extensions
        )
        
        if not docs:
            self.logger.warning("未找到任何文档")
            return
        
        # 准备文档和元数据
        contents = []
        metadatas = []
        
        for doc in docs:
            contents.append(doc['content'])
            
            # 合并元数据
            doc_metadata = doc['metadata'].copy()
            if metadata:
                doc_metadata.update(metadata)
            metadatas.append(doc_metadata)
        
        # 添加到知识库
        self.add_documents(contents, metadatas)
    
    def retrieve(
        self,
        query: str,
        context: str = "",
        top_k: int = 5,
        use_advanced: bool = True
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档

        Args:
            query: 查询文本
            context: 对话上下文
            top_k: 返回结果数量
            use_advanced: 是否使用高级RAG技术

        Returns:
            检索结果列表
        """
        tracer = get_tracer()
        tracer_span = tracer.start_span("rag.retrieve", {
            "rag.query": query,
            "rag.top_k": top_k,
            "rag.use_advanced": use_advanced,
        })
        tracer_span.__enter__()

        try:
            return self._do_retrieve(query, context, top_k, use_advanced, tracer)
        except Exception as e:
            tracer.record_exception(e)
            raise
        finally:
            tracer_span.__exit__(None, None, None)

    def _do_retrieve(self, query, context, top_k, use_advanced, tracer):
        """实际的检索逻辑"""
        self.logger.info(f"检索查询: {query}, 使用高级技术: {use_advanced}")

        queries = [query]

        if use_advanced and self.rag_config.get('query_rewrite', {}).get('enabled', True):
            # 路由器决定走哪个重写层级（Level 0–3），每条查询只走一层
            # use_deep=True 对应 rag_advanced intent，触发 Level 3（HyDE + 多查询）
            rewrite_result = self.rewrite_router.route_and_rewrite(
                query=query,
                context=context,
                use_deep=(use_advanced and self.rag_config.get('hyde', {}).get('enabled', True)),
            )
            queries = rewrite_result.queries
            self.logger.info(
                f"[RewriteRouter] level={rewrite_result.level.value} | "
                f"reason={rewrite_result.reason} | queries={len(queries)}"
            )
        
        # 去重
        queries = list(dict.fromkeys(queries))
        self.logger.info(f"总共 {len(queries)} 个查询")
        
        # 检索
        all_results = []
        for q in queries:
            # 生成查询嵌入
            query_embedding = self.embeddings.embed_query(q)
            
            # 向量检索
            initial_k = self.rag_config.get('rerank', {}).get('top_k', 10)
            vector_results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=initial_k
            )
            
            # 混合检索
            if self.rag_config.get('hybrid_search', {}).get('enabled', True):
                hybrid_results = self.hybrid_retriever.hybrid_search(
                    vector_results=vector_results,
                    query=q,
                    top_k=initial_k
                )
                all_results.extend(hybrid_results)
            else:
                # 只使用向量检索结果
                for i in range(len(vector_results['ids'][0])):
                    all_results.append({
                        'id': vector_results['ids'][0][i],
                        'document': vector_results['documents'][0][i],
                        'metadata': vector_results['metadatas'][0][i],
                        'score': 1 - vector_results['distances'][0][i]
                    })
        
        # 去重（基于ID）
        unique_results = {}
        for result in all_results:
            doc_id = result['id']
            if doc_id not in unique_results or result['score'] > unique_results[doc_id]['score']:
                unique_results[doc_id] = result
        
        results = list(unique_results.values())
        self.logger.info(f"去重后有 {len(results)} 个结果")
        
        # 重排序
        if use_advanced and self.rag_config.get('rerank', {}).get('enabled', True):
            final_k = self.rag_config.get('rerank', {}).get('final_k', top_k)
            results = self.reranker.simple_rerank(query, results, final_k)
            self.logger.info(f"重排序后返回 {len(results)} 个结果")
        else:
            # 按分数排序
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:top_k]
        
        # 扩展到父块
        expanded_results = self._expand_to_parent_chunks(results)

        tracer.set_span_attributes({
            "rag.total_queries": len(queries),
            "rag.results_count": len(expanded_results),
        })
        tracer.set_span_ok()

        return expanded_results
    
    def _expand_to_parent_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将子块扩展到父块"""
        expanded = []
        seen_parents = set()
        
        for result in results:
            parent_id = result.get('metadata', {}).get('parent_id')
            if parent_id and parent_id not in seen_parents:
                parent_text = result.get('metadata', {}).get('parent_text', result.get('document'))
                expanded.append({
                    **result,
                    'document': parent_text,
                    'is_expanded': True
                })
                seen_parents.add(parent_id)
            elif not parent_id:
                expanded.append(result)
        
        self.logger.debug(f"扩展到父块，结果数量: {len(expanded)}")
        return expanded
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        count = self.vector_store.get_collection_count()
        sample = self.vector_store.peek(limit=3)
        
        return {
            "document_count": count,
            "sample_documents": sample
        }
