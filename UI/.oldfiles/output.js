var test_context = document.getElementById('test_canvas').getContext('2d');

var image_data = test_context.createImageData(256, 256);


// mod 4
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
    // image_data.data[(x + y * 256) * 4] = r;
    // image_data.data[(x + y * 256) * 4 + 1] = r / 6;
    // image_data.data[(x + y * 256) * 4 + 2] = 0;
    // image_data.data[(x + y * 256) * 4 + 3] = 255;
  }
}

function createFig() {
  test_context.putImageData(image_data, 0, 0);
  // console.log(offsetX, offsetY, Number(inputElemDepth.value), Number(inputElemMag.value))
}