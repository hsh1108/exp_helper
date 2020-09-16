import torch
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


"""
save image using torchvision.
"""

def test():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dir = './samples'
    test_data = dsets.ImageFolder(dataset_dir, transform=transform) # Images should be contained in subdirectory.
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=True, num_workers=4)

    test_img = Variable(iter(test_loader).next()[0], requires_grad=True)
    pic = (test_img.data + 1) / 2.0
    torchvision.utils.save_image(pic, 'sample_img.jpg', nrow=1)

if __name__ == '__main__':
    test()