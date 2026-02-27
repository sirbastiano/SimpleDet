(() => {
  const path = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-link').forEach((link) => {
    const target = new URL(link.href, window.location.origin).pathname.split('/').pop();
    if (target === path) {
      link.classList.add('active');
    }
  });

  document.querySelectorAll('pre').forEach((block) => {
    const button = document.createElement('button');
    button.className = 'copy-btn';
    button.type = 'button';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      const text = block.innerText;
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = 'Copied';
        setTimeout(() => (button.textContent = 'Copy'), 1100);
      } catch (error) {
        button.textContent = 'Fail';
        setTimeout(() => (button.textContent = 'Copy'), 1100);
      }
    });
    block.appendChild(button);
  });
})();
