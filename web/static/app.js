/**
 * SmartAgent 前端交互逻辑
 */

// ================================================================
// 全局状态
// ================================================================
const state = {
    userId: '',
    username: '',
    sessionId: '',
    isSending: false,
};

// ================================================================
// API 请求封装
// ================================================================
async function api(url, options = {}) {
    const resp = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    if (!resp.ok && resp.status !== 200) {
        const err = await resp.json().catch(() => ({ detail: '请求失败' }));
        throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
}

// ================================================================
// 登录
// ================================================================
async function doLogin() {
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const errorEl = document.getElementById('login-error');
    const btn = document.getElementById('login-btn');

    if (!username || !password) {
        errorEl.textContent = '请输入用户名和密码';
        return;
    }

    btn.disabled = true;
    btn.textContent = '登录中...';
    errorEl.textContent = '';

    try {
        const data = await api('/api/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });

        state.userId = data.user_id;
        state.username = data.username;
        state.sessionId = data.session_id;

        document.getElementById('display-username').textContent = data.username;
        document.getElementById('login-page').style.display = 'none';
        document.getElementById('main-page').style.display = 'flex';

        loadSessions();
    } catch (e) {
        errorEl.textContent = e.message;
    } finally {
        btn.disabled = false;
        btn.textContent = '登 录';
    }
}

function doLogout() {
    state.userId = '';
    state.username = '';
    state.sessionId = '';
    document.getElementById('login-page').style.display = 'flex';
    document.getElementById('main-page').style.display = 'none';
    document.getElementById('messages').innerHTML = '';
    document.getElementById('password').value = '';
}

// ================================================================
// 会话管理
// ================================================================
async function loadSessions() {
    try {
        const data = await api(`/api/sessions?user_id=${state.userId}`);
        renderSessionList(data.sessions);

        // 加载当前会话历史
        if (state.sessionId) {
            loadHistory(state.sessionId);
        }
    } catch (e) {
        console.error('加载会话列表失败:', e);
    }
}

function renderSessionList(sessions) {
    const list = document.getElementById('session-list');
    if (!sessions.length) {
        list.innerHTML = '<div style="text-align:center;padding:20px;opacity:0.4;font-size:13px;">暂无历史对话</div>';
        return;
    }

    list.innerHTML = sessions.map(s => {
        const active = s.session_id === state.sessionId ? ' active' : '';
        const date = new Date(s.updated_at * 1000).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        return `
            <div class="session-item${active}" onclick="switchSession('${s.session_id}')" title="${s.title}">
                <span class="session-title">💬 ${escapeHtml(s.title)}</span>
                <span class="session-count">${s.message_count}</span>
                <button class="btn-delete-session" onclick="event.stopPropagation();deleteSession('${s.session_id}')" title="删除">🗑</button>
            </div>
        `;
    }).join('');
}

async function newSession() {
    try {
        const data = await api('/api/sessions/new', {
            method: 'POST',
            body: JSON.stringify({ user_id: state.userId }),
        });
        state.sessionId = data.session_id;
        clearMessages();
        showWelcome();
        updateChatTitle('新对话');
        loadSessions();
    } catch (e) {
        console.error('创建会话失败:', e);
    }
}

async function switchSession(sessionId) {
    if (sessionId === state.sessionId) return;
    try {
        await api(`/api/sessions/${sessionId}/resume`, {
            method: 'POST',
            body: JSON.stringify({ user_id: state.userId }),
        });
        state.sessionId = sessionId;
        loadSessions();
        loadHistory(sessionId);
    } catch (e) {
        console.error('切换会话失败:', e);
    }
}

async function deleteSession(sessionId) {
    if (!confirm('确认删除此对话？')) return;
    try {
        await api(`/api/sessions/${sessionId}?user_id=${state.userId}`, { method: 'DELETE' });
        if (sessionId === state.sessionId) {
            await newSession();
        }
        loadSessions();
    } catch (e) {
        console.error('删除会话失败:', e);
    }
}

async function loadHistory(sessionId) {
    try {
        const data = await api(`/api/sessions/${sessionId}/history?user_id=${state.userId}`);
        clearMessages();

        if (!data.messages.length) {
            showWelcome();
            updateChatTitle('新对话');
            return;
        }

        data.messages.forEach(m => appendMessage(m.role, m.content));

        // 更新标题
        const firstUser = data.messages.find(m => m.role === 'user');
        if (firstUser) {
            const title = firstUser.content.substring(0, 30) + (firstUser.content.length > 30 ? '...' : '');
            updateChatTitle(title);
        }

        scrollToBottom();
    } catch (e) {
        console.error('加载历史失败:', e);
    }
}

// ================================================================
// 发送消息
// ================================================================
async function sendMessage() {
    const input = document.getElementById('msg-input');
    const message = input.value.trim();
    if (!message || state.isSending) return;

    state.isSending = true;
    document.getElementById('send-btn').disabled = true;
    input.value = '';
    autoResizeInput();

    // 移除欢迎消息
    const welcome = document.querySelector('.welcome-msg');
    if (welcome) welcome.remove();

    // 显示用户消息
    appendMessage('user', message);

    // 显示打字动画
    const typingId = showTyping();

    try {
        const data = await api('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ user_id: state.userId, message }),
        });

        removeTyping(typingId);
        appendMessage('bot', data.response);

        // 更新标题（首条消息时）
        const msgs = document.querySelectorAll('.message');
        if (msgs.length <= 2) {
            const title = message.substring(0, 30) + (message.length > 30 ? '...' : '');
            updateChatTitle(title);
            loadSessions();
        }
    } catch (e) {
        removeTyping(typingId);
        appendMessage('bot', `⚠️ 发送失败: ${e.message}`);
    } finally {
        state.isSending = false;
        document.getElementById('send-btn').disabled = false;
        input.focus();
    }
}

// ================================================================
// 消息渲染
// ================================================================
function appendMessage(role, content) {
    const container = document.getElementById('messages');
    const isUser = role === 'user';

    const msgEl = document.createElement('div');
    msgEl.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatar = isUser ? '👤' : '🤖';
    const formatted = isUser ? escapeHtml(content) : formatBotMessage(content);

    msgEl.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-bubble">${formatted}</div>
    `;

    container.appendChild(msgEl);
    scrollToBottom();
}

function formatBotMessage(text) {
    // 简单的 markdown 渲染
    let html = escapeHtml(text);

    // 代码块 ```...```
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:13px;">$1</code>');
    // 粗体
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // 换行
    html = html.replace(/\n/g, '<br>');

    return html;
}

function showTyping() {
    const container = document.getElementById('messages');
    const id = 'typing-' + Date.now();
    const el = document.createElement('div');
    el.className = 'message bot';
    el.id = id;
    el.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="msg-bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    container.appendChild(el);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function clearMessages() {
    document.getElementById('messages').innerHTML = '';
}

function showWelcome() {
    document.getElementById('messages').innerHTML = `
        <div class="welcome-msg">
            <div class="welcome-icon">🤖</div>
            <h3>你好！我是 SmartAgent</h3>
            <p>我可以回答问题、检索知识库、调用工具完成任务。试试问我些什么吧！</p>
        </div>
    `;
}

function updateChatTitle(title) {
    document.getElementById('chat-title').textContent = title;
}

function scrollToBottom() {
    const container = document.getElementById('messages');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ================================================================
// 工具面板
// ================================================================
async function showToolsPanel() {
    document.getElementById('tools-modal').style.display = 'flex';
    try {
        const data = await api(`/api/tools?user_id=${state.userId}`);
        renderTools(data.tools, data.rate_limit_stats);
    } catch (e) {
        document.getElementById('tools-list').innerHTML = `<p>加载失败: ${e.message}</p>`;
    }
}

function renderTools(tools, stats) {
    const list = document.getElementById('tools-list');
    if (!tools.length) {
        list.innerHTML = '<p style="text-align:center;color:#999;">暂无注册工具</p>';
        return;
    }

    list.innerHTML = tools.map(t => {
        const badgeClass = t.source === 'mcp' ? 'mcp' : 'local';
        const status = t.enabled ? '✅' : '❌';
        return `
            <div class="tool-item">
                <span class="tool-badge ${badgeClass}">${t.source.toUpperCase()}</span>
                <div class="tool-info">
                    <div class="tool-name">${status} ${escapeHtml(t.name)}</div>
                    <div class="tool-desc">${escapeHtml(t.description)}</div>
                    <div class="tool-meta">分类: ${t.category} · 调用: ${t.call_count}次</div>
                </div>
            </div>
        `;
    }).join('');

    // 频率限制统计
    const info = document.getElementById('rate-limit-info');
    info.textContent = `📊 频率限制: 全局 ${stats.global_calls_last_minute}/${stats.global_limit_per_minute}/min · 会话总量 ${stats.session_total_calls}/${stats.session_limit}`;
}

function closeToolsPanel() {
    document.getElementById('tools-modal').style.display = 'none';
}

// ================================================================
// UI 辅助
// ================================================================
function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('collapsed');
}

function handleInputKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// 自动调整输入框高度
const input = document.getElementById('msg-input');
if (input) {
    input.addEventListener('input', autoResizeInput);
}

function autoResizeInput() {
    const el = document.getElementById('msg-input');
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 点击模态框外部关闭
document.addEventListener('click', (e) => {
    const modal = document.getElementById('tools-modal');
    if (e.target === modal) closeToolsPanel();
});

// Enter 快捷键登录
document.getElementById('password')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doLogin();
});
