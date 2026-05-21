// Copy BibTeX to clipboard.
const copyBtn = document.getElementById('copy-bibtex');
if (copyBtn) {
  copyBtn.addEventListener('click', async () => {
    const code = document.querySelector('#bibtex pre code');
    if (!code) return;

    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      const label = copyBtn.querySelector('span');
      const original = label.textContent;
      label.textContent = 'Copied!';
      setTimeout(() => (label.textContent = original), 1500);
    } catch (_) {
      // Fallback: select the text so the user can copy manually.
      const range = document.createRange();
      range.selectNodeContents(code);
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }
  });
}
