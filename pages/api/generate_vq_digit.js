import { vqModel, vqTransformerModel, device } from '../../serverState';
import { Response } from 'express';
import { Image } from 'image-js';
import { BytesIO } from 'bytes-io';

export default function handler(req, res) {
  if (req.method === 'GET') {
    try {
      if (!vqModel || !vqTransformerModel) {
        return res.status(500).send('VQ models not loaded');
      }

      const digit = parseInt(req.query.digit, 10);
      console.log(`Generating VQ image for digit ${digit}...`);

      // Generate image using VQ-Transformer and VQ-VAE
      vqTransformerModel.eval();
      with torch.no_grad() {
        const generatedImg = vqTransformerModel.generate(digit, vqModel, device);

        // Convert to numpy array
        const imgArray = generatedImg.cpu().squeeze().numpy();

        // Scale to [0, 255]
        const scaledImgArray = (imgArray * 255).astype(np.uint8);

        // Convert to PIL Image
        const outImg = Image.fromArray(scaledImgArray, { mode: 'L' });

        // Save as PNG in memory
        const buf = new BytesIO();
        outImg.save(buf, { format: 'PNG' });
        buf.seek(0);

        return res.status(200).send(buf.getvalue());
      }
    } catch (error) {
      console.error(`Error generating VQ image: ${error.message}`);
      return res.status(500).send(error.message);
    }
  } else {
    res.status(405).send('Method not allowed');
  }
}
