// Function to create the switcher button
function createSwitcherButton() {
    const button = document.createElement('button');
    button.style.position = 'absolute';
    button.style.top = '10px';
    button.style.right = '10px';
    button.style.padding = '8px';
    button.style.backgroundColor = '#4CAF50'; // Green
    button.style.color = 'white';
    button.style.border = 'none';
    button.style.borderRadius = '20px'; // Make it rounded
    button.style.cursor = 'pointer';
    button.style.zIndex = 1000;
    button.style.fontSize = '12px'; // Smaller font size

    // Check the current URL to determine which button to display
    if (window.location.href.includes(".com")) {
        button.innerText = '切换到中文'; // Switch to Chinese
        button.onclick = function () {
            const currentUrl = window.location.href;
            const newUrl = currentUrl.replace(".com", ".cn");
            window.location.href = newUrl; // Open in the same window
        };
    } else if (window.location.href.includes(".cn")) {
        button.innerText = 'Switch to English'; // Switch to English
        button.onclick = function () {
            const currentUrl = window.location.href;
            const newUrl = currentUrl.replace(".cn", ".com");
            window.location.href = newUrl; // Open in the same window
        };
    }

    // Append button to body only if it is not already added
    if (!document.body.contains(button)) {
        document.body.appendChild(button);
    }
}

// Wait for the page to load before creating the button
window.onload = createSwitcherButton;

// MutationObserver to detect dynamic content loading
const observer = new MutationObserver(createSwitcherButton);
observer.observe(document.body, { childList: true, subtree: true });
