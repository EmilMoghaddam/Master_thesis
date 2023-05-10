import torchvision

root = "http://groups.csail.mit.edu/vision/SUN/"
SUN = torchvision.datasets.SUN397(root=root, transform=None, target_transform=None, download=True)