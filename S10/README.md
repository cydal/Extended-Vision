#### Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 



# Tiny ImageNet 


* Contains 200 image classes
* 100,000 training images
* 10,000 validation images
* 10,000 test images
* Images Size 64×64


[![image.png](https://i.postimg.cc/fbzgRh64/image.png)](https://postimg.cc/BjzCmRSN)



## Image Augmentation


* Pad & Random Crop
* HorizontalFlip: P = 0.25
* Rotate: Limit = 15
* CoarseDropout: Holes = 1, 8x8
* RGBShift: P=0.25


```python
def get_train_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        train_transforms (Albumentation): Transform Object
    """

    train_transform = A.Compose([
                            A.Sequential([
                              A.PadIfNeeded(min_height=72, min_width=72),
                              A.RandomCrop(h, w),
                              A.HorizontalFlip(p=0.25),
                              A.Rotate(limit=15, p=0.5),
                              A.CoarseDropout(max_holes=1, max_height=8, max_width=8, fill_value=mu),
                              A.RGBShift(p=0.25),
                              A.Normalize(mean=(mu), 
                                          std=std),
                              ToTensorV2()
                            ], p=1)
    ])

    return(train_transform)

def get_val_transforms(h, w, mu, std):
    """
    Args:
        h (int): image height
        w (int):image width
        mu (array): image mean
        std (array): standard deviation
    Returns:
        val_transforms (Albumentation): Transform Object
    """
    val_transforms = A.Compose([
                            A.Resize(h, w, cv2.INTER_NEAREST),
                            A.Normalize(mean=(mu), 
                                        std=std),
                            ToTensorV2()
    ])

    return(val_transforms)
```


### LR Finder

```python
def lr_finder(model, optimizer, criterion, trainloader, device, end_lr, num_iter):
    """
    Args:
      model (torch.nn Model): 
      optimizer (optimizer) - Optimizer Object
      criterion (criterion) - Loss Function
      train_loader (DataLoader) - DataLoader Object
      device (str): cuda/CPU
      end_lr (float) - 
      num_iter (int) - Number of iterations 
    """

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=end_lr, num_iter=num_iter, step_mode="linear")
    lr_finder.plot()
```



[![image.png](https://i.postimg.cc/bwSpz1fc/image.png)](https://postimg.cc/xXQW5Jdt)


[![W-B-Chart-23-07-2021-21-11-54.png](https://i.postimg.cc/X7QkVyc8/W-B-Chart-23-07-2021-21-11-54.png)](https://postimg.cc/56QCgyrY)
[![W-B-Chart-23-07-2021-21-12-14.png](https://i.postimg.cc/T1cq5y8T/W-B-Chart-23-07-2021-21-12-14.png)](https://postimg.cc/m1tF0g2n)

[![W-B-Chart-23-07-2021-21-12-38.png](https://i.postimg.cc/nVqdtzPR/W-B-Chart-23-07-2021-21-12-38.png)](https://postimg.cc/c647RsZw)
[![W-B-Chart-23-07-2021-21-12-26.png](https://i.postimg.cc/qMLjzF02/W-B-Chart-23-07-2021-21-12-26.png)](https://postimg.cc/w71hr01B)
