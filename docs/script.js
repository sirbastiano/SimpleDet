(() => {
  const THEMES = ['theme-aurora', 'theme-midnight', 'theme-volcano'];
  const DEFAULT_THEME = 'theme-aurora';
  const STORAGE_THEME = 'simpledet-docs-theme';
  const path = window.location.pathname.split('/').pop() || 'index.html';

  const body = document.body;
  const storedTheme = window.localStorage ? window.localStorage.getItem(STORAGE_THEME) : null;
  const hasStoredTheme = THEMES.includes(storedTheme || '');

  const initTheme = hasStoredTheme ? storedTheme : DEFAULT_THEME;
  THEMES.forEach((theme) => body.classList.remove(theme));
  body.classList.add(initTheme);

  document.querySelectorAll('.theme-btn').forEach((button) => {
    if (button.dataset.theme === initTheme) {
      button.classList.add('active');
    }
    button.addEventListener('click', () => {
      const theme = button.dataset.theme;
      if (!THEMES.includes(theme)) {
        return;
      }
      THEMES.forEach((candidate) => body.classList.remove(candidate));
      body.classList.add(theme);
      document.querySelectorAll('.theme-btn').forEach((item) => item.classList.remove('active'));
      button.classList.add('active');
      if (window.localStorage) {
        window.localStorage.setItem(STORAGE_THEME, theme);
      }
    });
  });

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
