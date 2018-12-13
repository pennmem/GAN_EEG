import sys
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
import torch.utils.data as data
from torchvision import datasets,transforms
import torch
import numpy as np

data_dir = '/Users/tungphan/PycharmProjects/GAN_EEG/cycle_gan/datasets/'
dataset = 'apple2orange'

train_folder_a = data_dir + dataset + '/trainA/'
test_folder_a = data_dir + dataset + '/testA/'
train_folder_b = data_dir + dataset + '/trainB/'
test_folder_b = data_dir + dataset + '/testB/'



train_A_imglist = os.listdir(train_folder_a)
train_B_imglist = os.listdir(train_folder_b)




def plot_raw_imgs(imglist, dirname, idx=0):
    test_img = cv2.imread(os.path.join(dirname, imglist[idx]))
    print(test_img.shape)
    print(type(test_img))
    plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')


# plot_raw_imgs(train_B_imglist, train_folder_b, 10)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)



def cv_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #return Image.fromarray(img)
    return img

def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
#         if not os.path.isdir(d):
#             continue

        if is_image_file(d):
#             path = os.path.join(dir, d)
#             item = path#(path, class_to_idx[target])
            images.append(d)

    return images




class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=cv_loader):
#         classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

        return img


    def __len__(self):
        return len(self.imgs)



def get_iphone_dataset(batch_size, im_size=300, random_crop_size=256, testing = False):
    random_crop_size = 256
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.Resize(im_size),
         transforms.RandomCrop(random_crop_size),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    train_a_ = ImageFolder(train_folder_a, transform=None)
    train_b_ = ImageFolder(train_folder_b, transform=None)
    test_a_ = ImageFolder(test_folder_a, transform=None)
    test_b_ = ImageFolder(test_folder_b, transform=None)



    if testing:
        train_a_loader=torch.utils.data.DataLoader(train_a_, batch_size=batch_size,
                                               shuffle=True, num_workers=1)
        train_b_loader=torch.utils.data.DataLoader(train_b_, batch_size=batch_size,
                                               shuffle=True, num_workers=1)
    else:
        train_a_loader=torch.utils.data.DataLoader(test_a_, batch_size=batch_size,
                                               shuffle=True, num_workers=1)
        train_b_loader=torch.utils.data.DataLoader(test_b_, batch_size=batch_size,
                                               shuffle=True, num_workers=1)



    return train_a_loader, train_b_loader




def sample_images(epoch, batch_i, generator_A2B, generator_B2A):


    #os.makedirs('/Volumes/RHINO/home2/tungphan/GAN_EEG/images/%s' % dataset, exist_ok=True)
    r, c = 2, 3

    a_loader,b_loader = get_iphone_dataset(batch_size=1, testing=True)
    a_loader = iter(a_loader)
    b_loader = iter(b_loader)

    imgs_A = a_loader.next().numpy()
    imgs_B = b_loader.next().numpy()

    imgs_A = (imgs_A-128.0)/128.0
    imgs_B = (imgs_B-128.0)/128.0


    # Demo (for GIF)
    #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
    #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

    # Translate images to the other domain
    fake_B = generator_A2B.predict(imgs_A)
    fake_A = generator_B2A.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = generator_B2A.predict(fake_B)
    reconstr_B = generator_A2B.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1

    try:
        fig.savefig("/Volumes/RHINO/home2/tungphan/GAN_EEG/images/%s/%d_%d.png" % (dataset, epoch, batch_i))
    except:
        print("cannot save image")
    plt.close()


