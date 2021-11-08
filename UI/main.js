let target = document.getElementById("figJapan");

const inputElemDepth = document.getElementById('inputDepth');
const inputElemMag = document.getElementById('inputMag');

target.addEventListener("click", getPosition);

let offsetX = 0
let offsetY = 0

// >>> for get INPUT >>>
function getPosition(e) {
  offsetX = e.offsetX; // =>図形左上からのx座標
  offsetY = e.offsetY; // =>図形左上からのy座標

  offsetX = Math.floor(offsetX / 2)
  offsetY = Math.floor(offsetY / 2)

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
function createFig() {
  var test_context = document.getElementById('test_canvas').getContext('2d');
  var image_data = test_context.createImageData(256, 256);
  // data[x] in image_data means 
  // x mod 4 switch
  // 0:r
  // 1:g
  // 2:b
  // 3:alpha
  for (var y = 0; y < 256; y++) {
    for (var x = 0; x < 256; x++) {
      var r = random = Math.random() * 255;
      // var r = x;
      image_data.data[(x + y * 256) * 4] = 255;
      image_data.data[(x + y * 256) * 4 + 1] = 10;
      image_data.data[(x + y * 256) * 4 + 2] = 0;
      image_data.data[(x + y * 256) * 4 + 3] = r;
    }
  }

  test_context.putImageData(image_data, 0, 0);


  console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))
}
// <<< for OUTPUT<<<