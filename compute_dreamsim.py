from dreamsim import dreamsim
import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path1", type=str, required=True)
    parser.add_argument("--image_path2", type=str, required=True)
    # parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--cache_dir", type=str, default="/home/vmurugan/.cache/dreamsim"
    )
    args = parser.parse_args()

    assert os.path.exists(args.image_path1), "Image path 1 does not exist"
    assert os.path.exists(args.image_path2), "Image path 2 does not exist"
    # assert os.path.exists(os.path.dirname(args.output_path)), "Output path does not exist"

    file_names = os.listdir(args.image_path1)

    file_paths1 = [os.path.join(args.image_path1, i) for i in file_names]
    file_paths2 = [os.path.join(args.image_path2, i) for i in file_names]

    model, preprocess = dreamsim(pretrained=True, cache_dir=args.cache_dir)

    scores = []
    for p1, p2 in tqdm(zip(file_paths1, file_paths2), total=len(file_paths1)):
        img1 = preprocess(Image.open(p1)).to(model.device)
        img2 = preprocess(Image.open(p2)).to(model.device)
        score = model(img1, img2)
        scores.append(score.item())

    score = np.mean(scores)
    print(f"DreamSim Score: {score}")


if __name__ == "__main__":
    main()
