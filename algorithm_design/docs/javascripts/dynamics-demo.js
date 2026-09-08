(function() {
  function multiply(matrix, vector) {
    return [
      matrix[0] * vector[0] + matrix[1] * vector[1],
      matrix[2] * vector[0] + matrix[3] * vector[1]
    ];
  }

  function mount(root) {
    if (root.dataset.mounted === "true") return;
    root.dataset.mounted = "true";
    var matrix = root.dataset.k.split(",").map(Number);
    var initial = root.dataset.z0.split(",").map(Number);
    var input = root.querySelector("[data-horizon-input]");
    var horizon = root.querySelector("[data-horizon]");
    var state = root.querySelector("[data-state]");
    var operator = root.querySelector("[data-operator]");
    var canvas = root.querySelector("[data-dynamics-canvas]");
    var context = canvas.getContext("2d");

    function draw() {
      var steps = Number(input.value);
      var points = [initial.slice()];
      for (var index = 0; index < 32; index += 1) {
        points.push(multiply(matrix, points[points.length - 1]));
      }
      var visible = points.slice(0, steps + 1);
      var maxValue = Math.max.apply(null, points.map(function(point) {
        return Math.max(Math.abs(point[0]), Math.abs(point[1]));
      }));
      var scale = 105 / Math.max(maxValue, 1);
      var centerX = 450;
      var centerY = 160;
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = getComputedStyle(root).getPropertyValue("--demo-bg") || "#f5f8f7";
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.strokeStyle = "#cbd9d7";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(30, centerY);
      context.lineTo(870, centerY);
      context.moveTo(centerX, 20);
      context.lineTo(centerX, 300);
      context.stroke();
      context.strokeStyle = "#126d5c";
      context.lineWidth = 3;
      context.beginPath();
      visible.forEach(function(point, index) {
        var x = centerX + point[0] * scale;
        var y = centerY - point[1] * scale;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
      visible.forEach(function(point, index) {
        context.fillStyle = index === visible.length - 1 ? "#bd7a2f" : "#126d5c";
        context.beginPath();
        context.arc(centerX + point[0] * scale, centerY - point[1] * scale, index === visible.length - 1 ? 6 : 3, 0, 2 * Math.PI);
        context.fill();
      });
      horizon.value = String(steps);
      state.textContent = "z₍" + steps + "₎ = (" + visible[visible.length - 1].map(function(value) { return value.toFixed(3); }).join(", ") + ")";
      operator.textContent = "K = [" + matrix.map(function(value) { return value.toFixed(2); }).join(", ") + "]";
    }

    input.addEventListener("input", draw);
    draw();
  }

  function mountAll() {
    document.querySelectorAll(".koopman-demo").forEach(mount);
  }

  if (window.document$) document$.subscribe(mountAll);
  else document.addEventListener("DOMContentLoaded", mountAll);
})();
