window.MathJax = {
  loader: {
    load: ["[tex]/ams", "[tex]/boldsymbol"]
  },
  tex: {
    packages: {"[+]": ["ams", "boldsymbol"]},
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
