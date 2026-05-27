import os
import logging
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)


class DocumentLoader:
    """增强的文档加载器，支持多种文档格式"""
    
    # 文档加载器映射表
    DOCUMENT_LOADER_MAPPING = {
        ".pdf": (PDFPlumberLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".doc": (UnstructuredWordDocumentLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
        ".csv": (CSVLoader, {"encoding": "utf8"}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".markdown": (UnstructuredMarkdownLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".htm": (UnstructuredHTMLLoader, {}),
    }
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("初始化增强文档加载器")
        self.logger.info(f"支持的文档格式: {', '.join(self.DOCUMENT_LOADER_MAPPING.keys())}")
    
    def load_single_file(self, file_path: str) -> str:
        """
        加载单个文档文件
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            文档内容字符串
        """
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件扩展名
        ext = os.path.splitext(file_path)[1].lower()
        
        # 检查是否支持该文件格式
        if ext not in self.DOCUMENT_LOADER_MAPPING:
            self.logger.error(f"不支持的文档格式: {ext}")
            supported_formats = ', '.join(self.DOCUMENT_LOADER_MAPPING.keys())
            raise ValueError(f"不支持的文档格式: {ext}。支持的格式: {supported_formats}")
        
        self.logger.info(f"加载文档: {os.path.basename(file_path)} (格式: {ext})")
        
        try:
            # 获取对应的文档加载器
            loader_class, loader_args = self.DOCUMENT_LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            
            # 加载文档
            documents = loader.load()
            
            # 合并多页内容
            content = "\n".join([doc.page_content for doc in documents])
            
            self.logger.info(f"文档加载成功: {os.path.basename(file_path)}, 字符数: {len(content)}")
            self.logger.debug(f"文档内容预览: {content[:200]}...")
            
            return content
        except Exception as e:
            self.logger.error(f"加载文档失败 {file_path}: {e}")
            raise
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        加载目录中的所有文档
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归加载子目录
            file_extensions: 指定要加载的文件扩展名列表，None表示加载所有支持的格式
            
        Returns:
            文档列表，每个元素包含 {content: str, metadata: dict}
        """
        if not os.path.isdir(directory_path):
            self.logger.error(f"目录不存在: {directory_path}")
            raise NotADirectoryError(f"目录不存在: {directory_path}")
        
        self.logger.info(f"加载目录: {directory_path}, 递归: {recursive}")
        
        documents = []
        
        # 确定要处理的文件扩展名
        if file_extensions:
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                         for ext in file_extensions]
        else:
            extensions = list(self.DOCUMENT_LOADER_MAPPING.keys())
        
        self.logger.info(f"目标文件格式: {', '.join(extensions)}")
        
        # 遍历目录
        if recursive:
            for root, _, files in os.walk(directory_path):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    
                    if ext in extensions:
                        try:
                            content = self.load_single_file(file_path)
                            documents.append({
                                'content': content,
                                'metadata': {
                                    'source': filename,
                                    'file_path': file_path,
                                    'file_type': ext,
                                    'directory': os.path.dirname(file_path)
                                }
                            })
                        except Exception as e:
                            self.logger.warning(f"跳过文件 {filename}: {e}")
        else:
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                
                if os.path.isfile(file_path):
                    ext = os.path.splitext(filename)[1].lower()
                    
                    if ext in extensions:
                        try:
                            content = self.load_single_file(file_path)
                            documents.append({
                                'content': content,
                                'metadata': {
                                    'source': filename,
                                    'file_path': file_path,
                                    'file_type': ext,
                                    'directory': directory_path
                                }
                            })
                        except Exception as e:
                            self.logger.warning(f"跳过文件 {filename}: {e}")
        
        self.logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def load_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        加载多个文档文件
        
        Args:
            file_paths: 文档文件路径列表
            
        Returns:
            文档列表
        """
        self.logger.info(f"批量加载 {len(file_paths)} 个文档")
        
        documents = []
        for file_path in file_paths:
            try:
                content = self.load_single_file(file_path)
                documents.append({
                    'content': content,
                    'metadata': {
                        'source': os.path.basename(file_path),
                        'file_path': file_path,
                        'file_type': os.path.splitext(file_path)[1].lower()
                    }
                })
            except Exception as e:
                self.logger.warning(f"跳过文件 {file_path}: {e}")
        
        self.logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """获取支持的文件格式列表"""
        return list(DocumentLoader.DOCUMENT_LOADER_MAPPING.keys())
    
    @staticmethod
    def is_supported_format(file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in DocumentLoader.DOCUMENT_LOADER_MAPPING
