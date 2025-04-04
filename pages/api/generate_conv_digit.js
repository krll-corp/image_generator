import { convModel, device } from '../../serverState';
import { Response } from 'express';
import { Image } from 'image-js';
import { BytesIO } from 'bytes-io';

export default function handler(req, res) {
  if (req.method === 'GET') {
    try {
      const digit = parseInt(req.query.digit, 10);
      console.log(`Generating Conv image for digit ${digit}...`);

      // Convert digit to tensor
      const label = torch.tensor([digit], { device }).long();

      convModel.eval();
      with torch.no_grad() {
        let out = convModel(label);  // Return shape (28, 28) after necessary squeezes
        out = out.squeeze();
        if (out.dim() === 3) {
          out = out.squeeze(0);
        }

        // Check shape
        if (out.dim() !== 2 || out.shape !== [28, 28]) {
          throw new Error(`Expected tensor of shape (28, 28), got shape ${out.shape}`);
        }

        // Scale to [0, 255] (assuming model outputs are in [0,1])
        out = (out * 255.0).cpu().numpy().astype(np.uint8);

        // Convert to PIL Image
        const outImg = Image.fromArray(out, { mode: 'L' });

        // Save as PNG in memory
        const buf = new BytesIO();
        outImg.save(buf, { format: 'PNG' });
        buf.seek(0);

        return res.status(200).send(buf.getvalue());
      }
    } catch (error) {
      console.error(`Error generating conv image: ${error.message}`);
      return res.status(500).send(error.message);
    }
  } else {
    res.status(405).send('Method not allowed');
  }
}
