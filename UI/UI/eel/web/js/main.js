const canvasWidth = 512
const canvasHeight = 512
const bitSize = 64
const gridSize = canvasWidth / bitSize

let notRuninng = true

const longtitudeMax = 46
// const longtitudeMin = 30
const longtitudeSpan = 16
const latitudeMin = 128
const latitudeSpan = 18


const canvas = document.getElementById("test_canvas");
var test_context = document.getElementById('test_canvas').getContext('2d');
let imagePath = "../fig/japan.png";

canvas.style.border = "5px solid rgb(149, 247, 245)";
// canvas.style.border = "5px solid rgb(0, 202, 199)";

const inputElemDepth = document.getElementById('inputDepth');
const inputElemMag = document.getElementById('inputMag');

canvas.addEventListener("click", getPosition);

var offsetX = 0
var offsetY = 0
var datalist = []

const color = [
  "#005FFF",
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

colorRed = [
  "#F8E0E0",
  "#F6CECE",
  "#F5A9A9",
  "#F78181",
  "#FA5858",
  "#FE2E2E",
  "#FF0000",
  "#DF0101",
  "#B40404",
  "#8A0808",
  "#610B0B",
  "#3B0B0B"
]

const currentValueDepth = document.getElementById('currentDepth'); // 埋め込む先のspan要素  
const currentValueMag = document.getElementById('currentMag'); // 埋め込む先のspan要素

function pixelXtoLatitude(X) {
  return Math.round(latitudeMin + X * latitudeSpan / bitSize);
}
function pixelYtoLongtitude(Y) {
  return Math.round(longtitudeMax - Y * longtitudeSpan / bitSize);
}

// >>> for get INPUT >>>
function getPosition(e) {
  offsetX = e.offsetX;
  offsetY = e.offsetY;
  offsetX = Math.floor(offsetX / gridSize)
  offsetY = Math.floor(offsetY / gridSize)
  createFig(mode = "pin")
  console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))
  document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ")</p>";
  // document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + offsetX + "," + offsetY + ")</p>";
}

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



function setKanto() {
  offsetX = Math.round((139 - latitudeMin) * bitSize / latitudeSpan);
  offsetY = Math.round((longtitudeMax - 35) * bitSize / longtitudeSpan);
  inputElemDepth.value = 25;
  inputElemMag.value = 7.9;
  createFig(mode = "pin");
  document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ")</p>";
  setCurrentValue(inputElemDepth.value, inputElemMag.value); // ページ読み込み時に値をセット
}

function setHigashinihon() {
  offsetX = Math.round((142 - latitudeMin) * bitSize / latitudeSpan);
  offsetY = Math.round((longtitudeMax - 38) * bitSize / longtitudeSpan);
  inputElemDepth.value = 24;
  inputElemMag.value = 9;
  createFig(mode = "pin");
  document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ")</p>";
  setCurrentValue(inputElemDepth.value, inputElemMag.value); // ページ読み込み時に値をセット
}

function setNankai() {
  offsetX = Math.round((134 - latitudeMin) * bitSize / latitudeSpan);
  offsetY = Math.round((longtitudeMax - 33) * bitSize / longtitudeSpan);
  inputElemDepth.value = 35;
  inputElemMag.value = 9;
  createFig(mode = "pin");
  document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ")</p>";
  setCurrentValue(inputElemDepth.value, inputElemMag.value); // ページ読み込み時に値をセット
}


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
      async function runmain() {
        document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ") Running...</p>";
        return new Promise((resolve, reject) => {
          try {
            let server = "ntp.nict.jp";
            if (server.trim().length == 0) {
              console.log("Enter NTP server address!");
              return;
            }
            // ここでPython側の処理を実行
            // datalist = []
            eel.ask_python_from_js_get_result(server, offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value));
            console.log(datalist);
            if (!datalist.length) {
              console.log("Redo")
              createFig();
            }
            console.log("running...");
            resolve();
          } catch (e) {
            reject();
          }

        });
      }

      async function execRun() {
        await runmain();
        document.getElementById('currentXY').innerHTML = "<p>(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + ") Finished</p>";
        console.log("finished");
        test_context.fillText("(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + "), 深さ:" + inputElemDepth.value + "km, マグニチュード:" + inputElemMag.value, 10, 20)
        // test_context.fillText("(経度,緯度)=(" + pixelXtoLatitude(offsetX) + "," + pixelYtoLongtitude(offsetY) + "), Depth=" + inputElemDepth.value, ", Mag=" + inputElemMag.value, 0, 0)
        for (var x = 0; x < bitSize; x++) {
          for (var y = 0; y < bitSize; y++) {
            data_i = Number(datalist[y * bitSize + x])
            // if (true) {
            if (data_i > 0) {
              test_context.fillStyle = colorRed[data_i + 2];
              // test_context.fillStyle = color[color.length - data_i];
              // test_context.fillStyle = `rgb(${Math.floor(255 - 0.5 * x)}, ${Math.floor(255 - 0.5 * y)}, 0)`;
              test_context.fillRect(x * gridSize, y * gridSize, gridSize, gridSize);
              test_context.font = gridSize - 2 + 'px';
              test_context.fillStyle = "black"
              test_context.fillText(datalist[y * bitSize + x], x * gridSize, y * gridSize + gridSize)
            }
          }
        }
        datalist = []
      }
      execRun();
    }

    if (mode == "pin" || mode == "run") {
      // >>> create × on Map >>>
      let frameSize = (5 + Number(inputElemMag.value)) * 3 + 1
      const side = 5 + Number(inputElemMag.value)
      let pinColor = 255 * (1 - Number(inputElemDepth.value) / 2000)
      var image_data = test_context.createImageData(1, 1);
      // data[x] in image_data means
      // x mod 4 switch
      // 0:r
      // 1:g
      // 2:b
      // 3:alpha
      image_data.data[0] = pinColor;
      image_data.data[1] = 0;
      image_data.data[2] = 0;
      image_data.data[3] = 255;

      lineThickness = Math.min(3, frameSize / 4)
      for (var y = 0; y < frameSize; y++) {
        for (var x = 0; x < frameSize; x++) {
          if (Math.abs(x - y) <= lineThickness || Math.abs(x + y - frameSize) <= lineThickness) {
            test_context.putImageData(image_data, gridSize * offsetX + x - side, gridSize * offsetY + y - side);
          }
        }
      }


      // <<< create × on Map <<<
    }
  });
  image.src = imagePath;
  console.log("done")
}

createFig(mode = "init")

eel.expose(run_js_from_python);
function run_js_from_python(msg) {
  // datalist is defined in line 16 as a global list
  datalist = msg.split(",")
}

function saveCanvas(canvas_id) {
  // var canvas = document.getElementById(canvas_id);
  //アンカータグを作成
  var a = document.createElement('a');
  //canvasをJPEG変換し、そのBase64文字列をhrefへセット
  a.href = canvas.toDataURL('image/jpeg', 0.85);
  //ダウンロード時のファイル名を指定
  a.download = 'EarthquakeData' + 'X' + pixelXtoLatitude(offsetX) + 'Y' + pixelYtoLongtitude(offsetY) + 'Depth' + inputElemDepth.value + 'Mag' + inputElemMag.value + '.jpg';
  //クリックイベントを発生させる
  a.click();
}