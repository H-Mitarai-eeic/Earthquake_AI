const canvasWidth = 512
const canvasHeight = 512
const bitSize = 256
const gridSize = canvasWidth / bitSize
let target = document.getElementById("test_canvas");

const inputElemDepth = document.getElementById('inputDepth');
const inputElemMag = document.getElementById('inputMag');

target.addEventListener("click", getPosition);

var offsetX = 0
var offsetY = 0
var datalist = []

// >>> for get INPUT >>>
function getPosition(e) {
  offsetX = e.offsetX; // =>図形左上からのx座標
  offsetY = e.offsetY; // =>図形左上からのy座標

  offsetX = Math.floor(offsetX / 2)
  offsetY = Math.floor(offsetY / 2)
  createFig(mode = "pin")
  console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))
  document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ")</p>";
}

const color = ["#005FFF",
  "#136FFF",
  "#2C7CFF",
  "#4689FF",
  "#5D99FF",
  "#75A9FF",
  "#8EB8FF",
  "#A4C6FF",
  "#BAD3FF",
  "#D9E5FF",
  "#FFDBC9",
  "#FFC7AF",
  "#FFAD90",
  "#FF9872",
  "#FF8856",
  "#FF773E",
  "#FF6928",
  "#FF5F17",
  "#FF570D",
  "#FF4F02"]

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

var test_context = document.getElementById('test_canvas').getContext('2d');
const canvas = document.getElementById("test_canvas");
let imagePath = "../fig/japan.png";
// >>> for OUTPUT >>>
// function createFig(mode = "run") {
//   draw(mode);
// };
// data[x] in image_data means
// x mod 4 switch
// 0:r
// 1:g
// 2:b
// // 3:alpha
function createFig(mode = "run") {
  console.log("draw");
  const image = new Image();
  image.addEventListener("load", function () {
    // canvas.width = image.naturalWidth;
    // canvas.height = image.naturalHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    console.log("load!");

    if (mode == "run") {
      function runmain() {
        document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ") Running...</p>";
        return new Promise((resolve, reject) => {
          try {

            // var image_data = test_context.createImageData(gridSize, gridSize);
            // let server = document.getElementById("ntp").value;
            let server = "ntp.nict.jp";
            if (server.trim().length == 0) {
              document.getElementById("result").innerHTML = "Enter NTP server address!";
              return;
            }
            eel.ask_python_from_js_get_result(server);
            console.log(datalist);
            // console.log(typeof (datalist))
            if (!datalist.length) {
              console.log("Redo")
              createFig();
            }
            // ここでPython側の処理を実行
            console.log("running...");
            // const datalist = msg.split(",");

            for (var x = 0; x < bitSize; x++) {
              for (var y = 0; y < bitSize; y++) {
                data_i = Number(datalist[y * bitSize + x])
                if (data_i > 6) {
                  test_context.fillStyle = color[color.length - data_i];
                  // test_context.fillStyle = `rgb(${Math.floor(255 - 0.5 * x)}, ${Math.floor(255 - 0.5 * y)}, 0)`;
                  test_context.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
                }
              }
            }

            // var image_data = test_context.createImageData(gridSize, gridSize);
            // for (var y = 0; y < bitSize; y++) {
            //   for (var x = 0; x < bitSize; x++) {
            //     // var r = random = Math.random() * 255;
            //     // var r = x;
            //     data_i = Number(datalist[y * bitSize + x])
            //     if (x + y < 20) {
            //       console.log(data_i, typeof (data_i));
            //       console.log(image_data);
            //     }
            //     // if (data_i > 5) {
            //     //   image_data.fillStyle = "#fff";
            //     //   test_context.putImageData(image_data, gridSize * x, gridSize * y);
            //     // }
            //     if (data_i > 8) {
            //       for (var ix = 0; ix < gridSize; ix++) {
            //         for (var iy = 0; iy < gridSize; iy++) {
            //           image_data.data[4 * ix + 0] = (data_i) * 200;
            //           image_data.data[4 * ix + 1] = 0;
            //           image_data.data[4 * ix + 2] = 0;
            //           image_data.data[4 * ix + 3] = 255;
            //         }
            //       }
            //       test_context.putImageData(image_data, gridSize * x, gridSize * y);
            //     }
            //   }
            // }
            // document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ")</p>";
            // document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ") Finished</p>";
            resolve();
          } catch (e) {
            reject();
          }
        });
      }
      // runmain().then(() => {
      //   document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ") Finished</p>";
      //   console.log("finished");
      // }).catch(e => {
      //   console.log("error occured");
      // })
      async function execRun() {
        await runmain();
        document.getElementById('currentXY').innerHTML = "<p>(X,Y)=(" + offsetX + "," + offsetY + ") Finished</p>";
        console.log("finished");
      }
      execRun();
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


    }
  });
  image.src = imagePath;
  console.log("done")
}


createFig(mode = "init")


eel.expose(run_js_from_python);
function run_js_from_python(msg) {
  // document.getElementById("result").innerHTML = msg;
  datalist = msg.split(",")
}