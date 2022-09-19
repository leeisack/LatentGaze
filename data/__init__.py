import os
from data.MPII import MPII
from data.ETH import ETH
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader



class Data:
    def __init__(self, args):

        self.loader_train = None
        if not args.test_only:

            if args.data_train == "MPII_train":
                trainset = MPII(args.data_train)
            elif args.data_train == "ETH_train":
                trainset = ETH(args.data_train)

            self.loader_train = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )


        if args.data_test == "MPII_validation":
                testset = MPII(args.data_test)
        elif args.data_test == "ETH_validation":
                testset = ETH(args.data_test)
        
        self.loader_test = DataLoader(
            testset,
            batch_size=1,
            num_workers=1,
            shuffle=True,
            pin_memory=not args.cpu
        )

