import logging
from typing import List, Dict, Any, Tuple
import re
import hashlib


class DocumentProcessor:
    """文档处理器：支持父子分块"""
    
    def __init__(
        self,
        parent_chunk_size: int = 1000,
        child_chunk_size: int = 200,
        chunk_overlap: int = 50,
        logger: logging.Logger = None
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(
            f"初始化文档处理器 - 父块大小: {parent_chunk_size}, "
            f"子块大小: {child_chunk_size}, 重叠: {chunk_overlap}"
        )
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        将文本分割成块
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            overlap: 重叠大小
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= text_length:
                break
        
        return chunks
    
    def process_documents(
        self,
        documents: List[str],
        metadata_list: List[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str], Dict[str, str]]:
        """
        处理文档，生成父子分块
        
        Args:
            documents: 文档列表
            metadata_list: 元数据列表
            
        Returns:
            (子块列表, 元数据列表, ID列表, 父子映射)
        """
        self.logger.info(f"开始处理 {len(documents)} 个文档")
        
        child_chunks = []
        child_metadatas = []
        child_ids = []
        parent_child_map = {}
        
        for doc_idx, document in enumerate(documents):
            metadata = metadata_list[doc_idx] if metadata_list else {}
            
            # 生成父块
            parent_chunks = self._create_chunks(
                document,
                self.parent_chunk_size,
                self.chunk_overlap
            )
            
            self.logger.debug(f"文档 {doc_idx} 生成 {len(parent_chunks)} 个父块")
            
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                # 生成父块ID
                parent_id = self._generate_id(f"{doc_idx}_{parent_idx}_{parent_chunk}")
                
                # 从父块生成子块
                sub_chunks = self._create_chunks(
                    parent_chunk,
                    self.child_chunk_size,
                    self.chunk_overlap // 2
                )
                
                for child_idx, child_chunk in enumerate(sub_chunks):
                    child_id = self._generate_id(f"{parent_id}_{child_idx}_{child_chunk}")
                    
                    child_chunks.append(child_chunk)
                    child_ids.append(child_id)
                    
                    # 添加元数据
                    child_metadata = {
                        **metadata,
                        "parent_id": parent_id,
                        "parent_text": parent_chunk,
                        "doc_idx": doc_idx,
                        "parent_idx": parent_idx,
                        "child_idx": child_idx
                    }
                    child_metadatas.append(child_metadata)
                    
                    # 记录父子关系
                    parent_child_map[child_id] = parent_id
        
        self.logger.info(
            f"文档处理完成 - 生成 {len(child_chunks)} 个子块, "
            f"{len(set(parent_child_map.values()))} 个父块"
        )
        
        return child_chunks, child_metadatas, child_ids, parent_child_map
    
    def _generate_id(self, text: str) -> str:
        """生成文档ID"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
