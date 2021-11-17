const canvasWidth = 512
const canvasHeight = 512

let target = document.getElementById("test_canvas");

const inputElemDepth = document.getElementById('inputDepth');
const inputElemMag = document.getElementById('inputMag');

target.addEventListener("click", getPosition);

var offsetX = 0
var offsetY = 0


// >>> for get INPUT >>>
function getPosition(e) {
  offsetX = e.offsetX; // =>図形左上からのx座標
  offsetY = e.offsetY; // =>図形左上からのy座標

  offsetX = Math.floor(offsetX / 2)
  offsetY = Math.floor(offsetY / 2)
  createFig(mode = "pin")
  console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))
}


const currentValueDepth = document.getElementById('currentDepth'); // 埋め込む先のspan要素
const currentValueMag = document.getElementById('currentMag'); // 埋め込む先のspan要素

// 現在の値をspanに埋め込む関数
const setCurrentValue = (val1, val2) => {
  currentValueDepth.innerText = val1;
  currentValueMag.innerText = val2;
}

// inputイベント時に値をセットする関数
const rangeOnChange = (e) => {
  setCurrentValue(inputElemDepth.value, inputElemMag.value);
}

window.onload = () => {
  inputElemDepth.addEventListener('input', rangeOnChange); // スライダー変化時にイベントを発火
  inputElemMag.addEventListener('input', rangeOnChange); // スライダー変化時にイベントを発火
  setCurrentValue(inputElemDepth.value, inputElemMag.value); // ページ読み込み時に値をセット
}
// <<< for get INPUT <<<

// >>> for OUTPUT >>>
function createFig(mode = "run") {
  var test_context = document.getElementById('test_canvas').getContext('2d');
  const canvas = document.getElementById("test_canvas");
  let imagePath = "./fig/japan.png";
  draw(canvas, imagePath);
  // data[x] in image_data means 
  // x mod 4 switch
  // 0:r
  // 1:g
  // 2:b
  // // 3:alpha
  function draw(canvas, imagePath) {
    console.log("draw");
    const image = new Image();
    image.addEventListener("load", function () {
      // canvas.width = image.naturalWidth;
      // canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      console.log("load!");

      if (mode == "run") {
        var image_data = test_context.createImageData(1, 1);
        for (var y = 0; y < canvasHeight; y++) {
          for (var x = 0; x < canvasWidth; x++) {
            var r = random = Math.random() * 255;
            // var r = x;
            if (r > 240) {
              image_data.data[0] = r;
              image_data.data[1] = 10;
              image_data.data[2] = 0;
              image_data.data[3] = 255;
              test_context.putImageData(image_data, x, y);
            }
          }
        }
      }

      if (mode == "pin" || mode == "run") {
        let frameSize = (5 + Number(inputElemMag.value)) * 2 + 1
        const side = 5 + Number(inputElemMag.value)
        let pinColor = 255 * (1 - Number(inputElemDepth.value) / 1000)
        var image_data = test_context.createImageData(1, 1);
        image_data.data[0] = 0;
        image_data.data[1] = pinColor;
        image_data.data[2] = 255;
        image_data.data[3] = 255;

        lineThickness = Math.min(3, frameSize / 4)
        for (var y = 0; y < frameSize; y++) {
          for (var x = 0; x < frameSize; x++) {
            // var r = x;

            if (Math.abs(x - y) <= lineThickness || Math.abs(x + y - frameSize) <= lineThickness) {
              test_context.putImageData(image_data, 2 * offsetX + x - side, 2 * offsetY + y - side);
            }
          }
        }

        // let frameSize = (4 + Number(inputElemMag.value)) * 1.5
        // let pinColor = 255 * (1 - Number(inputElemDepth.value) / 1000)
        // var image_data = test_context.createImageData(frameSize, frameSize);
        // for (var y = 0; y < image_data.height; y++) {
        //   for (var x = 0; x < image_data.width; x++) {
        //     if (Math.abs(x - y) <= 3 || Math.abs(x + y - frameSize) <= 3) {

        //       image_data.data[(x + y * image_data.width) * 4] = 0;
        //       image_data.data[(x + y * image_data.width) * 4 + 1] = pinColor;
        //       image_data.data[(x + y * image_data.width) * 4 + 2] = 255;
        //       image_data.data[(x + y * image_data.width) * 4 + 3] = 255;
        //     }
        //     else {
        //       image
        //     }
        //   }
        // }
        // test_context.putImageData(image_data, 2 * offsetX, 2 * offsetY);
      }
    });
    image.src = imagePath;
  }
  console.log("done")
}

createFig(init = true)

// var test_context = document.getElementById('test_canvas').getContext('2d');
// var image_data = test_context.createImageData(256, 256);
// // data[x] in image_data means 
// // x mod 4 switch
// // 0:r
// // 1:g
// // 2:b
// // 3:alpha
// for (var y = 0; y < 256; y++) {
//   for (var x = 0; x < 256; x++) {
//     var r = random = Math.random() * 255;
//     // var r = x;
//     image_data.data[(x + y * 256) * 4] = 255;
//     image_data.data[(x + y * 256) * 4 + 1] = 10;
//     image_data.data[(x + y * 256) * 4 + 2] = 0;
//     image_data.data[(x + y * 256) * 4 + 3] = r;
//   }
// }

// test_context.putImageData(image_data, 0, 0);


// console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))}
// <<< for OUTPUT<<<

// const canvas = document.getElementById("test_canvas");
// const ctx = canvas.getContext("2d");

// ctx.font = "40px Times New Roman";
// ctx.fillText("Into", 0, 40);

// ctx.font = "40px Times New Roman";
// ctx.strokeText("the", 20, 60);

// ctx.font = "40px Times New Roman";
// ctx.fillText("Program", 40, 80);