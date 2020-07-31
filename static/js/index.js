window.onload = function () {
  var canvas = document.getElementById("drawing-board");
  paper.setup(canvas);
  var group = new paper.Group();
  var tool = new paper.Tool();
  var path;

  tool.onMouseDown = function (event) {
    path = new paper.Path();
    path.strokeColor = "black";
    path.add(event.point);
  };

  tool.onMouseDrag = function (event) {
    path.add(event.point);
  };

  tool.onMouseUp = function (event) {
    group.addChild(path);
    let img = canvas.toDataURL("image/png");
    document.getElementById("input-display").innerHTML = `<img src="${img}"/>`;
    predict()
  };

  document.getElementById("clear").onclick = clearBoard;
  function clearBoard() {
    group.removeChildren();
    document.getElementById("input-display").innerHTML = ``;
  }
  function predict() {
    let img = canvas.toDataURL("image/png");
    fetch(`/predict`, {
      method: "POST",
      headers:{
        "Content-Type": "image/png"
      },
      body: img,
    }).then((res) => {
      res.json().then((data) => {
        window.console.log(data);
      });
    });
  }
};
