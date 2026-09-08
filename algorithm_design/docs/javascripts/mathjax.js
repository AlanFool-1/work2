window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"], ["\\$", "\\$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

if (window.document$) {
  document$.subscribe(function() {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  });
}
