import logging
from typing import List, Dict, Any

from utils.llm_client import LLMClient


class SelfFix:
    """自我修正：验证和改进RAG结果"""
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_iterations: int = 2,
        logger: logging.Logger = None
    ):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"初始化Self-Fix，最大迭代: {max_iterations}")
    
    def verify_and_fix(
        self,
        query: str,
        answer: str,
        context: List[str],
        conversation_context: str = "",
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        验证答案质量并进行修正

        Args:
            query: 用户查询
            answer: 生成的答案
            context: 检索到的RAG上下文（商品文档等）
            conversation_context: 对话历史上下文（含用户已提供的预算、身份等信息）
            iteration: 当前迭代次数

        Returns:
            包含验证结果和修正后答案的字典
        """
        self.logger.info(f"Self-Fix验证，迭代 {iteration + 1}/{self.max_iterations}")

        # 验证答案（传入对话上下文，避免误判已知信息为缺失）
        issues = self._identify_issues(query, answer, context, conversation_context)

        if not issues or iteration >= self.max_iterations:
            self.logger.info(f"验证完成，发现 {len(issues)} 个问题")
            return {
                "answer": answer,
                "issues": issues,
                "fixed": False,
                "iterations": iteration
            }

        # 尝试修正
        self.logger.info(f"发现 {len(issues)} 个问题，尝试修正")
        fixed_answer = self._fix_answer(query, answer, context, issues, conversation_context)

        # 递归验证修正后的答案
        if iteration + 1 < self.max_iterations:
            return self.verify_and_fix(query, fixed_answer, context, conversation_context, iteration + 1)
        else:
            return {
                "answer": fixed_answer,
                "issues": issues,
                "fixed": True,
                "iterations": iteration + 1
            }
    
    def _identify_issues(
        self,
        query: str,
        answer: str,
        context: List[str],
        conversation_context: str = "",
    ) -> List[str]:
        """
        识别答案中的问题

        Args:
            query: 查询
            answer: 答案
            context: RAG检索文档
            conversation_context: 对话历史（含用户已提供的信息）

        Returns:
            问题列表
        """
        self.logger.debug("识别答案中的问题")

        context_text = "\n\n".join(context[:3])
        conv_section = f"\n\n对话历史（用户已提供的信息）:\n{conversation_context}" if conversation_context else ""

        prompt = f"""请评估以下答案的质量，识别可能存在的问题。

用户问题: {query}{conv_section}

知识库上下文:
{context_text}

生成的答案:
{answer}

**重要提示**：对话历史中用户已明确说明的信息（如预算、身份、场景等）不属于问题，不要列出"信息不足"或"需要澄清"类问题。

请识别以下方面的问题（如果没有问题，输出"无问题"）：
1. 答案是否准确回答了问题
2. 答案是否与知识库上下文一致，是否存在事实错误或捏造内容
3. 答案是否完整

如果有问题，请列出问题，每行一个。如果没有问题，只输出"无问题"："""
        
        try:
            result = self.llm_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            ).strip()
            
            if "无问题" in result or "没有问题" in result:
                return []
            
            issues = [line.strip() for line in result.split('\n') if line.strip()]
            self.logger.debug(f"识别到 {len(issues)} 个问题，问题列表“：{result}")
            
            return issues
        except Exception as e:
            self.logger.error(f"识别问题失败: {e}")
            return []
    
    def _fix_answer(
        self,
        query: str,
        answer: str,
        context: List[str],
        issues: List[str],
        conversation_context: str = "",
    ) -> str:
        """
        修正答案

        Args:
            query: 查询
            answer: 原答案
            context: RAG检索文档
            issues: 问题列表
            conversation_context: 对话历史（含用户已提供的信息）

        Returns:
            修正后的答案
        """
        self.logger.debug("修正答案")

        context_text = "\n\n".join(context[:3])
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        conv_section = f"\n\n对话历史（用户已提供的信息，修正时必须遵守）:\n{conversation_context}" if conversation_context else ""

        prompt = f"""请根据识别出的问题，改进以下答案。

用户问题: {query}{conv_section}

知识库上下文:
{context_text}

原答案:
{answer}

需要修正的问题:
{issues_text}

**输出要求**：直接输出修正后的最终答案，不要输出分析过程、改进思路或任何元评论。答案应直接面向用户："""
        
        try:
            fixed_answer = self.llm_client.generate(
                prompt=prompt,
                temperature=0.5,
                max_tokens=500
            ).strip()
            self.logger.info("答案修正完成")
            
            return fixed_answer
        except Exception as e:
            self.logger.error(f"修正答案失败: {e}")
            return answer
    
    def quick_verify(self, query: str, answer: str, context: List[str]) -> bool:
        """
        快速验证答案是否相关
        
        Args:
            query: 查询
            answer: 答案
            context: 上下文
            
        Returns:
            答案是否相关
        """
        # 简单的关键词匹配验证
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words) if query_words else 0
        
        is_relevant = relevance > 0.3
        self.logger.debug(f"快速验证：相关度 {relevance:.2f}, 是否相关: {is_relevant}")
        
        return is_relevant
