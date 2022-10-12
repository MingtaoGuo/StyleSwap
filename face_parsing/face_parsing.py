import torch 
import torchvision.transforms as transforms
from bisenet import BiSeNet
from PIL import Image 
from tqdm import tqdm 
import numpy as np 
import argparse
import os 


class FaceParsing:
    def __init__(self, model_path) -> None:
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    def predict(self, img_path):
        with torch.no_grad():
            img = Image.open(img_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--model_path", type=str, default="saved_models/79999_iter.pth")
    args = parser.parse_args()

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)
    parts_idx = np.array([1, 2, 3, 4, 5, 6, 10, 11, 12, 13])
    files = os.listdir(args.img_dir)
    fp = FaceParsing(args.model_path)

    to_one_hot = []
    for i in range(19):
        to_one_hot.append(np.ones([512, 512])*i)
    to_one_hot = np.dstack(to_one_hot)

    for i in tqdm(range(len(files))):
        parsing = fp.predict(args.img_dir + "/" + files[i])
        masks = np.float32(np.equal(parsing[..., None], to_one_hot))
        masks = np.sum(masks[..., parts_idx], axis=-1)
        Image.fromarray(np.uint8(masks*255)).resize([args.resolution, args.resolution]).save(args.res_dir + "/" + files[i])
