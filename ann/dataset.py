import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


class DayClassifierDataset(Dataset):
    def __init__(self, day_list, image_classes):
        self.days = []
        self.labels = []
        self.transforms = transforms.Normalize(0.5, 0.5)

        self.classes = list(set(image_classes))
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        for day, day_class in zip(day_list, image_classes):
            self.days.append(torch.from_numpy(day).float())
            label = self.class_to_label[day_class]
            self.labels.append(label)

    def __getitem__(self, index):
        return self.days[index], self.labels[index]

    def __len__(self):
        return len(self.days)
