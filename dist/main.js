(() => {
  const links = [...document.querySelectorAll('.site-nav a')];
  const sections = links
    .map((link) => document.querySelector(link.getAttribute('href')))
    .filter(Boolean);

  const setActive = (id) => {
    links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${id}`));
  };

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (visible) setActive(visible.target.id);
    }, { rootMargin: '-18% 0px -68% 0px', threshold: [0, .2, .6] });
    sections.forEach((section) => observer.observe(section));
  }

  const copyButton = document.querySelector('#copy-citation');
  const citation = document.querySelector('#citation');
  if (copyButton && citation) {
    copyButton.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(citation.textContent);
        copyButton.textContent = 'Copied';
        window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
      } catch {
        copyButton.textContent = 'Select text';
      }
    });
  }
})();
