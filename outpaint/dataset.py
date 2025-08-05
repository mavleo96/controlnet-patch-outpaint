import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset

class OutPaintDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = []
        with open(os.path.join(path, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.path, source_filename))
        target = cv2.imread(os.path.join(self.path, target_filename))

        if source is None or target is None:
            raise FileNotFoundError(f"Missing image at idx={idx}: {source_filename}, {target_filename}")

        # Convert BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize
        source = source.astype(np.float32) / 255.0              # [0, 1]
        target = (target.astype(np.float32) / 127.5) - 1.0      # [-1, 1]
        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == '__main__':
    dataset = OutPaintDataset('/data/vmurugan/datasets/test/train')
    print(len(dataset))

    item = dataset[0]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)   # Should print (512, 512, 3) - HWC format
    print(hint.shape)  # Should print (512, 512, 3) - HWC format
