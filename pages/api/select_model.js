import { availableModels, setSelectedModel } from '../../serverState';

export default function handler(req, res) {
  if (req.method === 'POST') {
    const { model_type } = req.body;

    if (availableModels[model_type]) {
      setSelectedModel(model_type);
      res.status(200).json({ message: `Selected model: ${model_type}` });
    } else {
      res.status(400).json({ message: `Model ${model_type} not available` });
    }
  } else {
    res.status(405).json({ message: 'Method not allowed' });
  }
}
