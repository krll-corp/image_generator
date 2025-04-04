import { vqModel, device } from '../../serverState';
import { Response } from 'express';
import { Image } from 'image-js';
import { BytesIO } from 'bytes-io';

export default function handler(req, res) {
  if (req.method === 'GET') {
    try {
      if (!vqModel) {
        return res.status(500).send('VQ-VAE model not loaded');
      }

      const digit = parseInt(req.query.digit, 10);
      console.log(`Generating VQ-VAE reconstruction for digit ${digit}...`);

      // Create a test dataset to get a real MNIST digit
      const transform = transforms.Compose([transforms.ToTensor()]);
      const testDataset = datasets.MNIST(root="./data", train=false, download=true, transform=transform);

      // Find examples of the requested digit
      const digitIndices = [];
      for (let i = 0; i < testDataset.length; i++) {
        if (testDataset[i][1] === digit) {
          digitIndices.push(i);
        }
      }

      if (digitIndices.length === 0) {
        return res.status(404).send(`No examples of digit ${digit} found in test set`);
      }

      // Pick a random example of this digit
      const idx = digitIndices[Math.floor(Math.random() * digitIndices.length)];
      const img = testDataset[idx][0];

      // Process through VQ-VAE (encode and decode)
      vqModel.eval();
      with torch.no_grad() {
        const imgTensor = img.unsqueeze(0).to(device);  // Add batch dimension
        const encodedIndices = vqModel.encode(imgTensor);
        const reconstructed = vqModel.decode(encodedIndices);

        // Convert to numpy and scale to [0, 255]
        const imgArray = (reconstructed.cpu().squeeze().numpy() * 255).astype(np.uint8);

        // Create PIL image
        const outImg = Image.fromArray(imgArray, { mode: 'L' });

        // Save as PNG in memory
        const buf = new BytesIO();
        outImg.save(buf, { format: 'PNG' });
        buf.seek(0);

        return res.status(200).send(buf.getvalue());
      }
    } catch (error) {
      console.error(`Error in VQ-VAE reconstruction: ${error.message}`);
      return res.status(500).send(error.message);
    }
  } else {
    res.status(405).send('Method not allowed');
  }
}
