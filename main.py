from src.data.how2sign_dataset import How2SignDataset
from torch.utils.data import DataLoader
from model import MultiStreamModel

def main():

    dataset = How2SignDataset("data/annotations/how2sign_train.csv")
    loader = DataLoader(dataset)

    model = MultiStreamModel()

    for video_name, sentence in loader:
        # print("sign_x:", sign_x.shape)
        # print("finger_x:", finger_x.shape)
        # print("lip_x:", lip_x.shape)
        # print("tgt_seq:", tgt_seq.shape)

        # logits = model(sign_x, finger_x, lip_x, tgt_seq)

        print("Output:", video_name, sentence)

if __name__ == "__main__":
    main()