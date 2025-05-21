import { useState } from 'react';

export default function Home() {
  const [digit, setDigit] = useState(0);
  const [model, setModel] = useState('pixel');
  const [imageSrc, setImageSrc] = useState(null);
  const [progress, setProgress] = useState(0);

  const handleGenerate = async () => {
    if (model === 'vq') {
      const response = await fetch(`/generate_vq_digit?digit=${digit}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageSrc(url);
    } else {
      const response = await fetch(`/generate_conv_digit?digit=${digit}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setImageSrc(url);
    }
  };

  const handleModelChange = (e) => {
    setModel(e.target.value);
  };

  const handleDigitChange = (e) => {
    setDigit(e.target.value);
  };

  return (
    <div className="container">
      <h1>MNIST Digit Generator</h1>
      <div className="form-group">
        <label htmlFor="digit">Digit (0-9):</label>
        <input
          type="number"
          id="digit"
          value={digit}
          onChange={handleDigitChange}
          min="0"
          max="9"
        />
      </div>
      <div className="form-group">
        <label htmlFor="model">Model:</label>
        <select id="model" value={model} onChange={handleModelChange}>
          <option value="pixel">PixelTransformer</option>
          <option value="moe">MoEPixelTransformer</option>
          <option value="conv">ConvGenerator</option>
          <option value="vq">VQ-Transformer</option>
          <option value="vq-vae">VQ-VAE Only</option>
        </select>
      </div>
      <button onClick={handleGenerate}>Generate</button>
      {imageSrc && <img src={imageSrc} alt="Generated Digit" />}
      {model === 'vq' && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }}></div>
        </div>
      )}
    </div>
  );
}
