#!/usr/bin/env python3
"""
SmartAgent 交互式对话程序

这个脚本提供了一个交互式命令行界面，可以与Agent进行多轮对话。
"""

import sys
import time
import getpass
from agent.agent import SmartAgent
from auth import AuthManager
from session import SessionManager


def print_header():
    """打印欢迎信息"""
    print("\n" + "="*80)
    print(" "*25 + "SmartAgent 交互式对话")
    print("="*80)
    print("\n💡 使用提示：")
    print("  - 直接输入问题进行对话")
    print("  - 输入 '/help' 查看所有命令")
    print("  - 输入 '/quit' 或 '/exit' 退出程序")
    print("  - 支持多轮对话，Agent会记住上下文")
    print("="*80 + "\n")


def login_flow(config_path: str = "./config/settings.yaml"):
    """
    登录流程：检查本地Token -> 未登录则要求输入用户名密码

    Returns:
        (auth_manager, token_info) 或 (None, None) 如果登录失败
    """
    from utils import ConfigLoader
    config = ConfigLoader(config_path).config
    auth_mgr = AuthManager(config=config)

    print("\n" + "="*60)
    print(" "*20 + "🔐 用户登录")
    print("="*60)

    # 尝试从本地Token免登录
    saved_username = _get_last_username(auth_mgr)
    if saved_username:
        token_info = auth_mgr.get_local_token(saved_username)
        if token_info:
            import datetime
            expires = datetime.datetime.fromtimestamp(token_info.expires_at)
            print(f"\n✅ 自动登录: {token_info.username} (Token有效至 {expires.strftime('%m-%d %H:%M')})")
            return auth_mgr, token_info

    # 需要登录
    max_attempts = 3
    for attempt in range(max_attempts):
        print()
        username = input("用户名: ").strip()
        if not username:
            continue
        password = getpass.getpass("密  码: ")

        token_info = auth_mgr.login(username, password)
        if token_info:
            print(f"\n✅ 登录成功！欢迎 {username}")
            return auth_mgr, token_info
        else:
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"\n❌ 用户名或密码错误，还剩 {remaining} 次机会")
            else:
                print("\n❌ 登录失败次数过多")

    return None, None


def _get_last_username(auth_mgr: AuthManager) -> str:
    """获取最近登录的用户名（从本地Token文件）"""
    for token_data in auth_mgr._tokens.values():
        return token_data.get('username')
    return None


def session_selection_flow(session_mgr: SessionManager, user_id: str) -> str:
    """
    会话选择流程：列出历史会话或创建新会话

    Returns:
        选中的 session_id 或 None（新会话）
    """
    sessions = session_mgr.list_sessions(user_id, limit=10)

    print("\n" + "="*60)
    print(" "*18 + "💬 会话管理")
    print("="*60)

    if sessions:
        print("\n📝 历史会话：")
        for i, s in enumerate(sessions, 1):
            print(f"  {i}. {s.to_display()}")
        print(f"\n  0. 创建新会话")
        print()

        while True:
            choice = input("请选择 (输入编号, 回车创建新会话): ").strip()
            if not choice or choice == '0':
                print("\n✨ 创建新会话")
                return None  # 新会话
            try:
                idx = int(choice)
                if 1 <= idx <= len(sessions):
                    selected = sessions[idx - 1]
                    print(f"\n🔄 恢复会话: {selected.title}")
                    return selected.session_id
                else:
                    print(f"  请输入 0-{len(sessions)} 之间的数字")
            except ValueError:
                print("  请输入有效编号")
    else:
        print("\n  暂无历史会话，将创建新会话")
        return None


def print_sessions(agent):
    """打印当前用户的会话列表"""
    if not agent.user_id:
        print("\n⚠️  未登录用户，无会话列表\n")
        return

    sessions = agent.session_manager.list_sessions(agent.user_id, limit=20)
    print("\n" + "="*60)
    print(f"💬 会话列表 (当前会话: {agent.session_id[:8]}...)")
    print("="*60)
    if not sessions:
        print("  暂无历史会话")
    else:
        for i, s in enumerate(sessions, 1):
            marker = " ◀ 当前" if s.session_id == agent.session_id else ""
            print(f"  {i}. {s.to_display()}{marker}")
    print("="*60)
    print("  💡 使用 /switch <编号> 切换会话，/new 创建新会话\n")


def print_help():
    """打印帮助信息"""
    print("\n" + "="*60)
    print("📖 可用命令：")
    print("="*60)
    print("  /help          - 显示此帮助信息")
    print("  /quit, /exit   - 退出程序")
    print("  /sessions      - 查看历史会话列表")
    print("  /switch <N>    - 切换到第N个历史会话")
    print("  /new           - 创建新会话")
    print("  /delete <N..>  - 删除指定会话（如 /delete 1 3 5）")
    print("  /delete_all    - 删除所有会话")
    print("  /clear         - 清空短期记忆")
    print("  /reset         - 重置会话")
    print("  /history [n]   - 显示最近n轮对话（默认10）")
    print("  /summary       - 显示长期记忆总结")
    print("  /info          - 显示知识库信息")
    print("  /mcp           - 显示MCP服务信息")
    print("  /metrics       - 显示Agent评估指标")
    print("  /logout        - 注销并退出")
    print("  /verbose on    - 开启详细日志")
    print("  /verbose off   - 关闭详细日志")
    print("="*60 + "\n")


def print_knowledge_base_info(agent):
    """打印知识库信息"""
    try:
        info = agent.get_knowledge_base_info()
        print("\n" + "="*60)
        print("📚 知识库信息：")
        print("="*60)
        print(f"  文档块数量: {info['document_count']}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ 获取知识库信息失败: {e}\n")


def print_mcp_info(agent):
    """打印MCP服务信息"""
    print("\n" + "="*60)
    print("📡 MCP服务信息：")
    print("="*60)
    
    services = agent.list_mcp_services()
    
    if services:
        for service in services:
            print(f"\n  服务名称: {service['name']}")
            print(f"  描述: {service.get('description', 'No description')}")
            print(f"  端点: {service.get('endpoint', 'N/A')}")
    else:
        print("\n  ℹ️  当前没有启用的MCP服务")
        print("  💡 在 config/mcp_servers.yaml 中配置MCP服务")
    
    print("="*60 + "\n")


def print_conversation_history(agent, n=10):
    """打印对话历史"""
    print("\n" + "="*60)
    print(f"📝 最近 {n} 轮对话：")
    print("="*60)
    
    history = agent.get_conversation_history(last_n=n)
    
    if not history:
        print("  暂无对话历史")
    else:
        for msg in history:
            role = "👤 用户" if msg['role'] == 'user' else "🤖 Agent"
            content = msg['content']
            print(f"\n{role}:")
            print(f"  {content}")
    
    print("="*60 + "\n")


def print_long_term_summaries(agent):
    """打印长期记忆总结"""
    print("\n" + "="*60)
    print("🧠 长期记忆总结：")
    print("="*60)
    
    summaries = agent.get_long_term_summaries(limit=5)
    
    if not summaries:
        print("  暂无长期记忆总结")
        print("  （每5轮对话会自动生成总结）")
    else:
        for i, summary in enumerate(summaries, 1):
            print(f"\n  [{i}] {summary['timestamp']}")
            print(f"  {summary['summary']}")
    
    print("="*60 + "\n")


def handle_command(command, agent, verbose):
    """
    处理命令
    
    Returns:
        (should_continue, new_verbose)
    """
    parts = command.strip().split()
    cmd = parts[0].lower()

    if cmd in ['/quit', '/exit']:
        print("\n👋 再见！感谢使用SmartAgent！\n")
        return False, verbose

    elif cmd in ['/clear_memory', '/清空记忆']:
        agent.clear_long_term_memory()

    elif cmd == '/help':
        print_help()
    
    elif cmd == '/clear':
        agent.clear_short_term_memory()
        print("\n✅ 短期记忆已清空\n")
    
    elif cmd == '/reset':
        agent.reset_session()
        print("\n✅ 会话已重置\n")

    elif cmd == '/sessions':
        print_sessions(agent)

    elif cmd == '/new':
        agent.reset_session()
        print(f"\n✨ 新会话已创建: {agent.session_id[:8]}...\n")

    elif cmd == '/switch':
        if not agent.user_id:
            print("\n⚠️  未登录用户，无法切换会话\n")
        elif len(parts) < 2:
            print("\n❌ 用法: /switch <编号>  (先用 /sessions 查看列表)\n")
        else:
            try:
                idx = int(parts[1])
                sessions = agent.session_manager.list_sessions(agent.user_id, limit=20)
                if 1 <= idx <= len(sessions):
                    target = sessions[idx - 1]
                    if agent.resume_session(target.session_id):
                        print(f"\n🔄 已切换到会话: {target.title}\n")
                        # 显示最近对话
                        history = agent.get_conversation_history(last_n=6)
                        if history:
                            print("📝 最近对话:")
                            for msg in history:
                                role = "👤" if msg['role'] == 'user' else "🤖"
                                content = msg['content'][:100] + ('...' if len(msg['content']) > 100 else '')
                                print(f"  {role} {content}")
                            print()
                    else:
                        print("\n❌ 切换失败\n")
                else:
                    print(f"\n❌ 编号超出范围，请先用 /sessions 查看\n")
            except ValueError:
                print("\n❌ 请输入有效编号\n")

    elif cmd == '/delete':
        if not agent.user_id:
            print("\n⚠️  未登录用户，无法删除会话\n")
        elif len(parts) < 2:
            print("\n❌ 用法: /delete <编号...>  (如 /delete 1 3 5，先用 /sessions 查看列表)\n")
        else:
            try:
                indices = [int(p) for p in parts[1:]]
                sessions = agent.session_manager.list_sessions(agent.user_id, limit=20)
                ids_to_delete = []
                invalid = []
                for idx in indices:
                    if 1 <= idx <= len(sessions):
                        ids_to_delete.append(sessions[idx - 1].session_id)
                    else:
                        invalid.append(str(idx))
                if invalid:
                    print(f"\n⚠️  编号 {', '.join(invalid)} 超出范围，已忽略")
                if ids_to_delete:
                    deleted = agent.delete_sessions(ids_to_delete)
                    print(f"\n🗑️  已删除 {deleted} 个会话\n")
                else:
                    print("\n❌ 无有效会话可删除\n")
            except ValueError:
                print("\n❌ 请输入有效编号（如 /delete 1 3 5）\n")

    elif cmd == '/delete_all':
        if not agent.user_id:
            print("\n⚠️  未登录用户，无法删除会话\n")
        else:
            confirm = input("\n⚠️  确认删除所有会话？此操作不可恢复 (y/n): ").strip().lower()
            if confirm in ('y', 'yes', '是'):
                deleted = agent.delete_all_sessions()
                print(f"\n🗑️  已删除全部 {deleted} 个会话，已创建新会话\n")
            else:
                print("\n❌ 已取消\n")

    elif cmd == '/logout':
        print("\n👋 已注销\n")
        return False, verbose
    
    elif cmd == '/history':
        n = int(parts[1]) if len(parts) > 1 else 10
        print_conversation_history(agent, n)
    
    elif cmd == '/summary':
        print_long_term_summaries(agent)
    
    elif cmd == '/info':
        print_knowledge_base_info(agent)
    
    elif cmd == '/mcp':
        print_mcp_info(agent)
    
    elif cmd == '/metrics':
        agent.agent_evaluator.print_metrics()

    elif cmd == '/tools':
        tools = agent.tool_manager.list_tools_brief()
        print("\n" + "="*60)
        print(f"🛠️  已注册工具 ({len(tools)} 个)：")
        print("="*60)
        for t in tools:
            status = "✅" if t['enabled'] else "❌"
            print(f"  {status} [{t['source']}|{t['category']}] {t['name']} - {t['description']}")
            if t['call_count'] > 0:
                print(f"      调用次数: {t['call_count']}")
        # 显示频率限制统计
        if hasattr(agent, 'rate_limiter'):
            stats = agent.rate_limiter.get_stats()
            print(f"\n📊 频率限制: 全局 {stats['global_calls_last_minute']}/{stats['global_limit_per_minute']}/min | "
                  f"会话总量 {stats['session_total_calls']}/{stats['session_limit']}")
        print("="*60 + "\n")

    elif cmd == '/verbose':
        if len(parts) > 1:
            if parts[1].lower() == 'on':
                verbose = True
                print("\n✅ 详细日志已开启\n")
            elif parts[1].lower() == 'off':
                verbose = False
                print("\n✅ 详细日志已关闭\n")
            else:
                print("\n❌ 用法: /verbose on 或 /verbose off\n")
        else:
            print(f"\n当前详细日志状态: {'开启' if verbose else '关闭'}\n")
    
    else:
        print(f"\n❌ 未知命令: {cmd}")
        print("💡 输入 /help 查看所有可用命令\n")
    
    return True, verbose


def main():
    """主函数"""
    print_header()

    # ========== 1. 登录流程 ==========
    auth_mgr, token_info = login_flow()
    if not token_info:
        print("\n退出程序")
        sys.exit(0)

    user_id = token_info.user_id
    current_token = token_info.token

    # ========== 2. 会话选择 ==========
    from utils import ConfigLoader
    config = ConfigLoader().config
    session_mgr = SessionManager(config=config)
    selected_session_id = session_selection_flow(session_mgr, user_id)

    # ========== 3. 初始化Agent ==========
    print("\n🚀 正在初始化SmartAgent...")
    try:
        agent = SmartAgent(
            user_id=user_id,
            username=token_info.username,
            session_id=selected_session_id,
        )
        print("✅ SmartAgent 初始化完成！\n")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请检查：")
        print("  1. Ollama是否正在运行 (ollama serve)")
        print("  2. 模型是否已下载 (ollama pull llama3.2)")
        print("  3. 配置文件是否正确\n")
        sys.exit(1)

    # 如果是恢复的会话，显示最近对话
    if selected_session_id:
        history = agent.get_conversation_history(last_n=6)
        if history:
            print("📝 最近对话:")
            for msg in history:
                role = "👤 你" if msg['role'] == 'user' else "🤖 Agent"
                content = msg['content'][:120] + ('...' if len(msg['content']) > 120 else '')
                print(f"  {role}: {content}")
            print()

    # 显示初始信息
    print_knowledge_base_info(agent)

    verbose = True

    # 主对话循环
    print("💬 开始对话（输入 /help 查看命令）\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            # 检查是否是命令
            if user_input.startswith('/'):
                should_continue, verbose = handle_command(user_input, agent, verbose)
                if not should_continue:
                    break
                continue
            
            # 正常对话
            print()  # 空行
            response = agent.chat(user_input, verbose=verbose)
            print(f"\n🤖 Agent: {response}\n")
            
            # 刷新输出缓冲区并等待后台日志完成
            sys.stdout.flush()
            sys.stderr.flush()
            time.sleep(0.4)  # 给后台操作（如保存记忆）时间完成
            
            print()  # 空行
            
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！\n")
            break
        except EOFError:
            print("\n\n👋 再见！\n")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("💡 输入 /help 查看帮助，或输入 /quit 退出\n")


if __name__ == "__main__":
    main()
