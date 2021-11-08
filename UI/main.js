let target = document.getElementById("figJapan");

const inputElemDepth = document.getElementById('inputDepth');
const inputElemMag = document.getElementById('inputMag');

target.addEventListener("click", getPosition);

function getPosition(e) {
  let offsetX = e.offsetX; // =>図形左上からのx座標
  let offsetY = e.offsetY; // =>図形左上からのy座標

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


