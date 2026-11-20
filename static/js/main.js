// Demo tabs: show one case at a time, pause the others; copy the BibTeX.
(function () {
  const tabs = Array.from(document.querySelectorAll('.tab'));
  const demos = Array.from(document.querySelectorAll('.demo'));
  function show(i) {
    tabs.forEach(t => {
      const on = t.dataset.case === String(i);
      t.classList.toggle('active', on);
      t.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    demos.forEach(d => {
      const on = d.dataset.case === String(i);
      d.hidden = !on;
      const v = d.querySelector('video');
      if (!on && v) v.pause();
    });
  }
  tabs.forEach(t => t.addEventListener('click', () => show(t.dataset.case)));

  const copy = document.querySelector('.copy');
  if (copy) {
    copy.addEventListener('click', async () => {
      const text = document.querySelector('.bib code').innerText;
      try {
        await navigator.clipboard.writeText(text);
        copy.textContent = 'Copied';
      } catch (e) {
        copy.textContent = 'Select & copy';
      }
      setTimeout(() => (copy.textContent = 'Copy'), 1600);
    });
  }
})();
