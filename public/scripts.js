document.addEventListener("DOMContentLoaded", function () {
  const digitInput = document.getElementById("digitInput");
  const generateBtn = document.getElementById("generateBtn");
  const modelSelector = document.getElementById("modelSelector");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const progressContainer = document.getElementById("progress-container");
  const progressFill = document.getElementById("progress-fill");
  const log = document.getElementById("log");

  function updateProgressBar(progress) {
    progressFill.style.width = `${progress}%`;
  }

  function logMessage(message) {
    const p = document.createElement("p");
    p.textContent = message;
    log.appendChild(p);
  }

  function selectModel() {
    const model = modelSelector.value;
    fetch("/select_model", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model_type: model }),
    })
      .then((response) => response.json())
      .then((data) => {
        logMessage(data.message);
      })
      .catch((error) => {
        logMessage(`Error: ${error}`);
      });
  }

  function generateDigit() {
    const digit = digitInput.value;
    const model = modelSelector.value;

    if (model === "conv") {
      fetch(`/generate_conv_digit?digit=${digit}`)
        .then((response) => response.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const img = new Image();
          img.onload = function () {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            URL.revokeObjectURL(url);
          };
          img.src = url;
        })
        .catch((error) => {
          logMessage(`Error: ${error}`);
        });
    } else if (model === "vq") {
      const eventSource = new EventSource(`/stream_digit?digit=${digit}`);
      let imageData = ctx.createImageData(canvas.width, canvas.height);
      let pixelIndex = 0;

      eventSource.onmessage = function (event) {
        if (event.data.startsWith("token:")) {
          const [_, tokenIndex, progress] = event.data.split(":");
          updateProgressBar(progress);
        } else {
          const pixelValue = parseInt(event.data);
          const x = pixelIndex % canvas.width;
          const y = Math.floor(pixelIndex / canvas.width);
          const index = (y * canvas.width + x) * 4;
          imageData.data[index] = pixelValue;
          imageData.data[index + 1] = pixelValue;
          imageData.data[index + 2] = pixelValue;
          imageData.data[index + 3] = 255;
          pixelIndex++;

          if (pixelIndex === canvas.width * canvas.height) {
            ctx.putImageData(imageData, 0, 0);
            eventSource.close();
            updateProgressBar(100);
          }
        }
      };

      eventSource.onerror = function (error) {
        logMessage(`Error: ${error}`);
        eventSource.close();
      };
    } else {
      fetch(`/stream_digit?digit=${digit}`)
        .then((response) => {
          const reader = response.body.getReader();
          let imageData = ctx.createImageData(canvas.width, canvas.height);
          let pixelIndex = 0;

          function read() {
            reader.read().then(({ done, value }) => {
              if (done) {
                ctx.putImageData(imageData, 0, 0);
                return;
              }

              const pixelValue = value[0];
              const x = pixelIndex % canvas.width;
              const y = Math.floor(pixelIndex / canvas.width);
              const index = (y * canvas.width + x) * 4;
              imageData.data[index] = pixelValue;
              imageData.data[index + 1] = pixelValue;
              imageData.data[index + 2] = pixelValue;
              imageData.data[index + 3] = 255;
              pixelIndex++;

              read();
            });
          }

          read();
        })
        .catch((error) => {
          logMessage(`Error: ${error}`);
        });
    }
  }

  generateBtn.addEventListener("click", generateDigit);
  modelSelector.addEventListener("change", selectModel);
});
