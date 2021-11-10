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

var canvas = document.getElementById('test-canvas');
if (canvas != null) {
  // do here.  
  (function () {
    var context = canvas.getContext('2d');

    var srcs = [
      'fig/japan.png',
    ];
    var images = [];
    for (var i in srcs) {
      images[i] = new Image();
      images[i].src = srcs[i];
    }

    var loadedCount = 1;
    for (var i in images) {
      images[i].addEventListener('load', function () {
        if (loadedCount == images.length) {
          var x = 0;
          var y = 0;
          for (var j in images) {
            context.drawImage(images[j], x, y, 150, 100);
            x += 50;
            y += 70;
          }
        }
        loadedCount++;
      }, false);
    }
    console.log("hoge")
  })();
} else {
  console.log("null")
  // report the error.  
}




// create fig

window.onload = () => {
  // #image1に画像を描画
  drawImage1();

  // #image2にテキストを描画
  drawImage2();

  // 「+」ボタンを押したら合成
  document.querySelector("#btn-concat").addEventListener("click", () => {
    concatCanvas("#concat", ["#image1", "#image2"]);
  });

  // 「消しゴム」ボタンを押したらクリア
  document.querySelector("#btn-eraser").addEventListener("click", () => {
    eraseCanvas("#concat");
  });

};

/**
 * [onload] うな重の画像を描画
 */
function drawImage1() {
  const Unaju = new Image();
  Unaju.src = "image/unajyu.png";
  Unaju.onload = () => {
    const canvas = document.querySelector("#image1");
    const ctx = canvas.getContext("2d");
    ctx.drawImage(Unaju, 0, 0, canvas.width, canvas.height);
  }
}

/**
 * [onload] テキスト「うな重」を描画
 */
function drawImage2() {
  const canvas = document.querySelector("#image2");
  const ctx = canvas.getContext("2d");
  ctx.font = "32px serif";
  ctx.fillStyle = "Red";
  ctx.fillText("うな重", 45, 150);
}

/**
 * Canvas合成
 *
 * @param {string} base 合成結果を描画するcanvas(id)
 * @param {array} asset 合成する素材canvas(id)
 * @return {void}
 */
async function concatCanvas(base, asset) {
  const canvas = document.querySelector(base);
  const ctx = canvas.getContext("2d");

  for (let i = 0; i < asset.length; i++) {
    const image1 = await getImagefromCanvas(asset[i]);
    ctx.drawImage(image1, 0, 0, canvas.width, canvas.height);
  }
}

/**
 * Canvasをすべて削除する
 *
 * @param {string} target 対象canvasのid
 * @return {void}
 */
function eraseCanvas(target) {
  const canvas = document.querySelector(target);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/**
 * Canvasを画像として取得
 *
 * @param {string} id  対象canvasのid
 * @return {object}
 */
function getImagefromCanvas(id) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const ctx = document.querySelector(id).getContext("2d");
    image.onload = () => resolve(image);
    image.onerror = (e) => reject(e);
    image.src = ctx.canvas.toDataURL();
  });
}