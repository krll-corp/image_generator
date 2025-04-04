import { availableModels, selectedModel, device } from '../../serverState';
import { Response } from 'express';
import { torch } from 'torch-js';

export default function handler(req, res) {
  if (req.method === 'GET') {
    try {
      const digit = parseInt(req.query.digit, 10);
      const modelName = selectedModel;

      console.log(`Streaming ${modelName} image for digit ${digit}...`);

      if (!availableModels[modelName]) {
        return res.status(400).send(`Model ${modelName} is not available.`);
      }

      const pixelStream = async function* () {
        try {
          if (modelName === 'moe' && moeModel) {
            const generator = moeModel.generate_digit_stream(digit);
            for await (const pixel of generator) {
              const pixelValue = Math.floor(pixel * 255 / 9);
              yield `data: ${pixelValue}\n\n`;
              await new Promise(resolve => setTimeout(resolve, 5));
            }
          } else if (modelName === 'pixel' && pixelModel) {
            const generator = pixelModel.generate_digit_stream(digit);
            for await (const pixel of generator) {
              const pixelValue = Math.floor(pixel * 255 / 9);
              yield `data: ${pixelValue}\n\n`;
              await new Promise(resolve => setTimeout(resolve, 5));
            }
          } else if (modelName === 'vq' && vqTransformerModel && vqModel) {
            const generator = vqTransformerModel.generate_token_stream(digit, device);
            const tokens = [];
            for await (const token of generator) {
              tokens.push(token);
              const progress = Math.floor((tokens.length * 100) / 49);
              yield `data: token:${tokens.length}:${progress}\n\n`;
              await new Promise(resolve => setTimeout(resolve, 10));
            }
            if (tokens.length === 49) {
              const tokenTensor = torch.tensor(tokens, { dtype: 'int64', device }).reshape([1, 7, 7]);
              const decodedImg = vqModel.decode(tokenTensor);
              const imgArray = decodedImg.cpu().squeeze().numpy().map(v => Math.floor(v * 255));
              for (const pixel of imgArray) {
                yield `data: ${pixel}\n\n`;
                await new Promise(resolve => setTimeout(resolve, 1));
              }
            }
          } else {
            yield `data: Error: Invalid model selected or model not available.\n\n`;
          }
        } catch (error) {
          console.error(`Error in pixel stream: ${error.message}`);
          yield `data: Error: ${error.message}\n\n`;
        }
      };

      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const stream = pixelStream();
      const reader = stream.getReader();

      const push = async () => {
        const { done, value } = await reader.read();
        if (done) {
          res.end();
        } else {
          res.write(value);
          push();
        }
      };

      push();
    } catch (error) {
      console.error(`Error in stream_digit: ${error.message}`);
      res.status(500).send(error.message);
    }
  } else {
    res.status(405).send('Method not allowed');
  }
}
