renderMathInElement(document.querySelector('main'), {
  delimiters: [
    { left: '\\[', right: '\\]', display: true },
    { left: '\\(', right: '\\)', display: false },
  ],
  throwOnError: true,
  strict: 'error',
});
