// content.js
function createSwitcherButton() {
    const button = document.createElement('button');
    button.innerText = 'Switch to English'; // 默认文本
    button.style.position = 'fixed';
    button.style.bottom = '50px'; // 调整按钮位置，确保与 feedback 按钮在同一高度
    button.style.right = '80px'; // 紧贴反馈按钮的右边
    button.style.padding = '2px'; // 缩小按钮的 padding
    button.style.fontSize = '10px'; // 缩小字体
    button.style.backgroundColor = '#4CAF50';
    button.style.color = 'white';
    button.style.border = 'none';
    button.style.borderRadius = '3px'; // 缩小圆角
    button.style.cursor = 'pointer';
    button.style.zIndex = 1000;

    // 判断当前网址是 .com 还是 .cn，并设置相应的按钮文本和跳转逻辑
    if (window.location.hostname.includes("leetcode.com")) {
        button.innerText = '切换到中文'; // 切换到中文
        button.onclick = function () {
            const currentUrl = window.location.href;
            const newUrl = currentUrl.replace("leetcode.com", "leetcode.cn");
            window.open(newUrl, '_blank'); // 在新标签页中打开新链接
        };
    } else if (window.location.hostname.includes("leetcode.cn")) {
        button.innerText = 'Switch to English'; // 切换到英文
        button.onclick = function () {
            const currentUrl = window.location.href;
            const newUrl = currentUrl.replace("leetcode.cn", "leetcode.com");
            window.open(newUrl, '_blank'); // 在新标签页中打开新链接
        };
    }

    // 只在页面上创建一个按钮
    if (!document.body.contains(button)) {
        document.body.appendChild(button);
    }
}

// 页面加载时创建按钮
window.onload = createSwitcherButton;
