import { useState, useEffect } from 'react';

export default function Home() {
  const [digit, setDigit] = useState(7);
  const [model, setModel] = useState('pixel');
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [log, setLog] = useState('');
  const [imageData, setImageData] = useState(null);

  useEffect(() => {
    if (imageData) {
      const canvas = document.getElementById('canvas');
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0);
      };
      img.src = imageData;
    }
  }, [imageData]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    setLog('Generating...');
    setProgress(0);

    try {
      if (model === 'conv') {
        const response = await fetch(`/api/generate_conv_digit?digit=${digit}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const blob = await response.blob();
        setImageData(URL.createObjectURL(blob));
        setLog('Generated!');
      } else if (model === 'vq' || model === 'vq-vae') {
        const endpoint = model === 'vq-vae' ? 
          `/api/generate_vq_vae_digit?digit=${digit}` : 
          `/api/stream_digit?digit=${digit}`;
        
        if (model === 'vq-vae') {
          const response = await fetch(endpoint);
          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }
          const blob = await response.blob();
          setImageData(URL.createObjectURL(blob));
          setLog('Generated!');
        } else {
          const eventSource = new EventSource(endpoint);
          const canvas = document.getElementById('canvas');
          const ctx = canvas.getContext('2d');
          const imageData = ctx.createImageData(28, 28);
          let pixelCounter = 0;

          eventSource.onmessage = (event) => {
            const data = event.data;

            if (data.startsWith('Error:')) {
              setLog(data);
              eventSource.close();
              setIsGenerating(false);
              return;
            }

            if (data.startsWith('token:')) {
              const parts = data.split(':');
              const tokenNum = parseInt(parts[1]);
              const progress = parseInt(parts[2]);
              setProgress(progress);
              setLog(`Generating tokens: ${tokenNum}/49 (${progress}%)`);
              return;
            }

            const pixelValue = parseInt(data);
            if (isNaN(pixelValue)) {
              console.error('Invalid pixel value:', data);
              return;
            }

            const x = pixelCounter % 28;
            const y = Math.floor(pixelCounter / 28);
            const idx = (y * 28 + x) * 4;
            imageData.data[idx] = pixelValue;
            imageData.data[idx + 1] = pixelValue;
            imageData.data[idx + 2] = pixelValue;
            imageData.data[idx + 3] = 255;

            pixelCounter++;

            if (x === 27 || pixelCounter === 28 * 28) {
              ctx.putImageData(imageData, 0, 0);

              if (pixelCounter >= 28 * 28) {
                eventSource.close();
                setLog('Generation complete!');
                setIsGenerating(false);
              }
            }
          };

          eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            eventSource.close();
            setLog('Error in streaming!');
            setIsGenerating(false);
          };
        }
      } else {
        const eventSource = new EventSource(`/api/stream_digit?digit=${digit}`);
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(28, 28);
        let index = 0;

        eventSource.onmessage = (event) => {
          const data = event.data;
          if (data.startsWith('Error:')) {
            setLog(data);
            eventSource.close();
            setIsGenerating(false);
            return;
          }

          const pixelValue = parseInt(data);
          if (isNaN(pixelValue)) {
            console.error('Invalid pixel value:', data);
            return;
          }

          imageData.data[index] = pixelValue;
          imageData.data[index + 1] = pixelValue;
          imageData.data[index + 2] = pixelValue;
          imageData.data[index + 3] = 255;
          index += 4;

          if (index % (28 * 4) === 0) {
            ctx.putImageData(imageData, 0, 0);
          }

          if (index >= 28 * 28 * 4) {
            eventSource.close();
            setLog('Generation complete!');
            setIsGenerating(false);
          }
        };

        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          eventSource.close();
          setLog('Error in streaming!');
          setIsGenerating(false);
        };
      }
    } catch (error) {
      console.error('Error:', error);
      setLog(`Error generating image: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleModelChange = async (event) => {
    const selectedModel = event.target.value;
    setModel(selectedModel);

    await fetch('/api/select_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_type: selectedModel }),
    });
  };

  return (
    <div>
      <h2>Conditional MNIST Generator</h2>
      <p>Enter a digit (0-9) to generate:</p>
      <input
        type="number"
        value={digit}
        min="0"
        max="9"
        onChange={(e) => setDigit(e.target.value)}
        style={{ width: '60px' }}
      />
      <button onClick={handleGenerate} disabled={isGenerating}>
        Generate
      </button>
      <select value={model} onChange={handleModelChange} disabled={isGenerating}>
        <option value="pixel">PixelTransformer</option>
        <option value="moe">MoEPixelTransformer</option>
        <option value="conv">ConvGenerator</option>
        <option value="vq">VQ-Transformer</option>
        <option value="vq-vae">VQ-VAE Only</option>
      </select>
      <canvas id="canvas" width="28" height="28" style={{ width: '280px', height: '280px', border: '1px solid #ccc', imageRendering: 'pixelated', background: '#fff' }}></canvas>
      {model === 'vq' || model === 'vq-vae' ? (
        <div className="progress-bar" style={{ display: 'block', height: '20px', backgroundColor: '#f0f0f0', borderRadius: '5px', marginTop: '10px' }}>
          <div className="progress-fill" style={{ height: '100%', backgroundColor: '#4CAF50', borderRadius: '5px', width: `${progress}%`, transition: 'width 0.1s' }}></div>
        </div>
      ) : null}
      <div id="log" style={{ marginTop: '10px', whiteSpace: 'pre-wrap', fontSize: '14px', color: '#666' }}>{log}</div>
    </div>
  );
}
