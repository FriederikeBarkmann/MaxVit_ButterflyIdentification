import os
import psutil
import argparse
from functools import partial
import time, datetime
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter, OrderedDict
import torch
import torchvision  # Used in metaprogramming part
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed


# Parse command line arguments
parser = argparse.ArgumentParser(description='Butterflies with DDP',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default="resnet152",
                    help='Pretrained model to be finetuned')
parser.add_argument('--model-path', type=str, default="", required=False,
                    help='Finetuned model to be super-finetuned')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--optimizer', type=str, default="SGD",
                    help='Optimizer to use')
parser.add_argument('--weight-decay', type=float, default=1e-5,
                    help='Value of weight decay for the optimizer')
parser.add_argument('--scheduler', type=str, default="StepLR",
                    help='Learning rate scheduler to use')
parser.add_argument('--oversampling', type=int, default=0,
                    help='Whether using oversampling or class weights')
parser.add_argument('--checkpointing', type=int, default=0,
                    help='Whether to do checkpointing after each epoch')
parser.add_argument('--data-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory with butterfly dataset and indices')
parser.add_argument('--results-dir', type=str, default=os.environ.get("SCRATCH"),
                    help='Directory to store results and checkpoints')
parser.add_argument('--checkpoint', type=str, default="", required=False,
                    help='Checkpoint to continue training from')
args = parser.parse_args()


SLURM_JOB_ID = os.environ['SLURM_JOB_ID']


# Set CPU-GPU bindings for LUMI and LEONARDO
def set_cpu_affinity(local_rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    LEONARDO_GPU_CPU_map = {
        # A mapping from GPUs to the closest CPU cores in a LEOANRDO BOOSTER node
        # Note that all 4 GPUs are closer to the first NUMA node with CPU cores 0-15
        # See https://wiki.u-gov.it/confluence/display/SCAIUS/Booster+Section
        # However seems irrelevant to do on LEONARDO
        0: [0, 16, 1, 17, 2, 18, 3, 19],
        1: [4, 20, 5, 21, 6, 22, 7, 23],
        2: [8, 24, 9, 25, 10, 26, 11, 27],
        3: [12, 28, 13, 29, 14, 30, 15, 31],
    }
    LEONARDO_GPU_CPU_map = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
    }
    if (os.environ['MASTER_ADDR'].startswith("lrdn")):
        cpu_list = LEONARDO_GPU_CPU_map[local_rank]
    elif (os.environ['MASTER_ADDR'].startswith("nid")):
        cpu_list = LUMI_GPU_CPU_map[local_rank]
    #print(f"{local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)


def worker(args):
    # Keep reproducible (Accelerate sets all seeds)
    set_seed(42)

    # Folders for results and checkpoints
    results_folder = os.path.join(args.results_dir,
        f"{args.model}", SLURM_JOB_ID + "_" +
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        #datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Use ProjectConfiguration
    project_config= ProjectConfiguration(project_dir=results_folder,
                                         automatic_checkpoint_naming=True)

    # Initialize HF accelerator
    accelerator = Accelerator(project_config=project_config)

    # Check number of processes
    if accelerator.is_main_process:
        print(f"Accelerator processes: {accelerator.num_processes}\n")

    # Map GPUs to CPUs (positive effect only for LUMI)
    if (os.environ['MASTER_ADDR'].startswith("nid")):
        set_cpu_affinity(accelerator.local_process_index)


    # ======== Data Preparation ========


    # Transformations of training data to avoid overfitting
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size = 224, scale = (0.5, 1), ratio = (0.8, 1.2)),
        v2.RandomHorizontalFlip(p = 0.3),
        v2.RandomVerticalFlip(p = 0.3),
        v2.RandomPerspective(distortion_scale = 0.2, p = 0.4),
        v2.RandomRotation(degrees = 50, expand = False),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transformations of validation data (only resizes and crops images and normalizes)
    val_transform = v2.Compose([
        v2.Resize(size = 224),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Connect dataset
    data_dir = os.path.join(args.data_dir, "Schmetterlinge_sym")
    dataset = ImageFolder(root = data_dir)
    # Load both training and validation but not test subset (indices) of the data
    train_val_idx = np.load(os.path.join(args.data_dir,
        "preparation", "train_val_idx.npy"))


    # Class names
    classes = dataset.classes
    #accelerator.print(f"Classes: {classes}")
    # Number of classes
    n_classes = len(classes)  # -> Used for final classification layer
    #accelerator.print(f"Number of classes: {n_classes}")
    # Number of examples per class -> used for train-test-split
    targets = [dataset.targets[i] for i in train_val_idx]
    #accelerator.print(f"Targets: {targets[:15]}...")
    #accelerator.print("Examples per class: {}".format(Counter(targets)))


    # Split indices into training and validation sets (randomly)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size = 0.2,
        shuffle = True,
        stratify = targets  # Ensure similar number of examples per class
    )

    # Save training and validation set indices for reproducibility
    #np.save(os.path.join(args.data_dir, "preparation", "train_idx.npy"),
    #        train_idx)
    #np.save(os.path.join(args.data_dir, "preparation", "val_idx.npy"),
    #        val_idx)
    # Can save also number of examples per class for later use
    #np.save(os.path.join(args.data_dir, "preparation", "targets.npy"),
    #        np.array(targets))

    # Set number of muliprocessing workers for the data loaders
    num_workers = int(os.environ['OMP_NUM_THREADS'])


    # Load training data
    train_data = ImageFolder(root = data_dir, transform = train_transform)
    train_data = Subset(train_data, train_idx)

    # Legacy code to do oversampling before every training run
    if False:
        sample_startTime = time.time()
        #accelerator.print("Oversampling training data.\n")
        if accelerator.is_main_process:
            print("Oversampling training data.\n")
        # Count the number of images per class (replace with targets?)
        labels = [label for image, label in train_data]
        counts = Counter(labels)
        accelerator.print(counts)
        # Get the weight for each class (inverse value of the number of images in each class)
        class_weights_dict = dict(zip(counts.keys(),
                    [1/weights for weights in list(counts.values())]))
        # Assign the weights to each sample in the unbalanced dataset
        sample_weights = [class_weights_dict.get(i) for i in labels]
        # Total number of samples to be drawn (same as length of training dataset)
        # num_samples = len(train_dl.dataset)
        # Oversample minority classes with a random sampler
        train_sampler = torch.utils.data.WeightedRandomSampler(
                            weights = sample_weights,
                            num_samples = len(train_idx),
                            replacement = True)
        sample_endTime = time.time()
        accelerator.print("Done. Time used to sample training data: {:.2f}s".format(sample_endTime-sample_startTime))
        # Data loader
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                sampler = train_sampler,
                drop_last = True,
                pin_memory=True,
                num_workers = num_workers)

    # Load indices with oversampled minority classes
    if args.oversampling:
        if accelerator.is_main_process:
            print("Using oversampled training data.\n")
        indices = torch.load(os.path.join(args.data_dir,
            "preparation", "oversampler_indices.pth"))
        # Sampler to load indices of minority-oversampled dataset
        train_sampler = SubsetRandomSampler(indices)
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                sampler = train_sampler,
                drop_last = True,  # higher troughput
                pin_memory=True,
                num_workers = num_workers)

    else:
        # Data loader without weighted sampling
        if accelerator.is_main_process:
            print("Using class weights.\n")
        train_dl = DataLoader(train_data,
                batch_size = args.batch_size,
                drop_last = True,
                pin_memory=True,
                num_workers = num_workers)

    # Load valiation data with fewer trafos (and without reweighting)
    val_data = ImageFolder(root = data_dir,
            transform = val_transform)
    val_data = Subset(val_data, val_idx)
    val_dl = DataLoader(val_data,
            batch_size = args.batch_size,
            drop_last = True,
            num_workers = num_workers,
            pin_memory = True)


    # ======== Training ========

    startTime = time.time()
    if accelerator.is_main_process:
        print("Starting training.")

    # Set the pre-trained model
    # exec(f"model = torchvision.models.{args.model}(weights='DEFAULT')")  # exec troubles
    model = eval(f"torchvision.models.{args.model}(weights='DEFAULT')")

    # Adapt the last layer to classes of the dataset for finetuning
    if ("resne" in args.model or "regne" in args.model):
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, n_classes))
    elif (args.model.startswith("densenet")):
        if (args.model.endswith("201")):
            model.classifier=nn.Linear(1920, n_classes)
        elif (args.model.endswith("169")):
            model.classifier=nn.Linear(1664, n_classes)
        elif (args.model.endswith("161")):
            model.classifier=nn.Linear(2208, n_classes)
        elif (args.model.endswith("121")):
            model.classifier=nn.Linear(1024, n_classes)
    elif (args.model.startswith("vgg")):
        num_ftrs = 512*7*7
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_classes),
            nn.LogSoftmax(dim=1))
    elif (args.model.startswith("efficientnet")):
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.1, inplace=True),
            nn.Linear(1280, n_classes),)
    elif (args.model.startswith("vit")):
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if (args.model.startswith("vit_b")):
            hidden_dim=768
        elif (args.model.startswith("vit_l")):
            hidden_dim=1024
        elif (args.model.startswith("vit_h")):
            # Need to load other than default weights for images of size 224
            model = torchvision.models.vit_h_14(weights='IMAGENET1K_SWAG_LINEAR_V1')
            hidden_dim=1280
        heads_layers["head"] = nn.Linear(hidden_dim, n_classes)
        model.heads = nn.Sequential(heads_layers)
    elif (args.model.startswith("swin")):
        if (args.model.endswith("_t")):
            embed_dim=96
        elif (args.model.endswith("_s")):
            embed_dim=96
        elif (args.model.endswith("_b")):
            embed_dim=128
        num_features = embed_dim * 2 ** 3
        model.head = nn.Linear(num_features, n_classes)
    elif (args.model.startswith("maxvit")):
        block_channels=[64, 128, 256, 512]
        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(block_channels[-1]),
            nn.Linear(block_channels[-1], block_channels[-1]),
            nn.Tanh(),
            nn.Linear(block_channels[-1], n_classes, bias=False)
        )
    elif (args.model.startswith("convnext")):
        if (args.model.endswith("tiny")):
            lastblock_input_channels = 768
        elif (args.model.endswith("small")):
            lastblock_input_channels = 768
        elif (args.model.endswith("base")):
            lastblock_input_channels = 1024
        elif (args.model.endswith("large")):
            lastblock_input_channels = 1536
        class LayerNorm2d(nn.LayerNorm):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                x = x.permute(0, 3, 1, 2)
                return x
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        model.classifier = nn.Sequential(
            norm_layer(lastblock_input_channels),
            nn.Flatten(1),
            nn.Linear(lastblock_input_channels, n_classes))


    # Optimizer
    base_lr = args.base_lr * accelerator.num_processes  # Scale with devices
    #print(f"Scaled base learning rate: {base_lr}\n")

    if (args.optimizer == "Adam"):
        opt = torch.optim.Adam(model.parameters(), lr = base_lr,
                                weight_decay = args.weight_decay)
    elif (args.optimizer == "AdamW"):
        opt = torch.optim.AdamW(model.parameters(), lr = base_lr,
                                weight_decay = args.weight_decay)
    elif (args.optimizer == "RMSprop"):
        opt = torch.optim.RMSprop(model.parameters(), lr = base_lr,
                                weight_decay = args.weight_decay)
    elif (args.optimizer == "Adagrad"):
        opt = torch.optim.Adagrad(model.parameters(), lr = base_lr,
                                weight_decay = args.weight_decay)
    elif (args.optimizer == "Adadelta"):
        opt = torch.optim.Adadelta(model.parameters(), lr = base_lr,
                                weight_decay = args.weight_decay)
    elif (args.optimizer == "ASGD"):
        opt = torch.optim.ASGD(model.parameters(), lr = base_lr,
                               weight_decay = args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr = base_lr,
                              momentum=0.9)

    if (args.scheduler == "StepLR"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer = opt,
                                                    step_size = 2,
                                                    gamma = 0.75)
    elif (args.scheduler == "ReduceLROnPlateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = opt, # reduces learning when loss stops decreasing
                mode = "min",
                factor = 0.75, # factor by which the learning rate is scaled
                patience = 2) # number of epochs without improvement until learning rate decreases
    elif (args.scheduler == "CosineAnnealingLR"):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer = opt,
                T_max = args.epochs, # number of epochs for one cycle
                eta_min = 1e-8)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer = opt,
                                                        factor=1,
                                                        total_iters=50000)
    accelerator.register_for_checkpointing(scheduler)


    # Cross-Entropy Loss
    if not args.oversampling:
        # Estimation of class weights using original train+val dataset
        class_weights = compute_class_weight("balanced",
                classes=np.arange(n_classes),
                y=np.array(targets))
        # Put class weiths to device manually; loss moved at backward method
        class_weights = torch.from_numpy(class_weights).float().to(accelerator.device)
        lossFN = nn.CrossEntropyLoss(weight=class_weights)
    else:
        lossFN = nn.CrossEntropyLoss()

    # Synchronize batch normalization
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Load model weights, a checkpoint or start fresh
    if args.model_path:
        #model = accelerator.unwrap_model(model)
        model.load_state_dict(torch.load(args.model_path), strict=False)
        #model = load_checkpoint_and_dispatch(model, args.model_path)
        model, train_dl, val_dl, opt, scheduler = accelerator.prepare(
                model, train_dl, val_dl, opt, scheduler)
        if accelerator.is_main_process:
            print(f"Using model saved at {args.model_path}.\n")
    elif args.checkpoint:
        model, train_dl, val_dl, opt, scheduler = accelerator.prepare(
                model, train_dl, val_dl, opt, scheduler)
        accelerator.load_state(args.checkpoint)
        if accelerator.is_main_process:
            print(f"Using checkpoint {args.checkpoint}.\n")
    else:
        if accelerator.is_main_process:
            print(f"Fresh finetuning of {args.model}.\n")
        #accelerator.save_state(safe_serialization=False)
        model, train_dl, val_dl, opt, scheduler = accelerator.prepare(
                model, train_dl, val_dl, opt, scheduler)


    # Calculate number of training and validation steps
    trainSteps = math.ceil(len(train_dl.dataset) / args.batch_size)
    valSteps = math.ceil(len(val_dl.dataset) / args.batch_size)

    # Store training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "train_time": []
    }


    # Training loop
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        trainSamples = 0
        valSamples = 0

        # Training loop
        for (x, y) in train_dl:
            # Forward pass and training loss
            pred = model(x)   # Make a prediction for x
            loss = lossFN(pred, y)  # Calculate the loss

            # Backpropagation and update weights
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # Update total training loss and number of correct predictions
            preds, ys = accelerator.gather((pred, y))
            cor_preds = preds.argmax(1) == ys
            trainCorrect += cor_preds.sum()
            losses = accelerator.gather(loss)
            totalTrainLoss += losses.sum()
            trainSamples += ys.shape[0]  # Actual samples (last batch cut)

        # Validation loop
        with torch.no_grad():
            model.eval()
            for (x, y) in val_dl:
                pred = model(x)
                preds, ys = accelerator.gather((pred, y))
                cor_preds = preds.argmax(1) == ys
                valCorrect += cor_preds.sum()
                loss = lossFN(pred, y)
                losses = accelerator.gather(loss)
                totalValLoss += losses.sum()
                valSamples += ys.shape[0]

        # Calculate average losses
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps

        # Calculate training and validation accuracy
        trainCorrect = trainCorrect / trainSamples
        valCorrect = valCorrect / valSamples

        endTime = time.time()
        epochTime = endTime-startTime
        epochTime_clean = time.strftime('%H:%M:%S', time.gmtime(epochTime))

        # Update training history per epoch
        history['train_loss'].append(avgTrainLoss.cpu().detach().numpy())
        history['train_acc'].append(trainCorrect)
        history['val_loss'].append(avgValLoss.cpu().detach().numpy())
        history['val_acc'].append(valCorrect)
        history['train_time'].append(endTime-startTime)

        # Print some statistics
        if accelerator.is_main_process:
            print("Epoch: {}/{}".format(epoch+1, epochs))
            print(f"Training time: {epochTime_clean}")
            #print("Learning rate: {}".format(scheduler.get_last_lr()[0]))
            print("Train loss: {:.4f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
            print("Validation loss: {:.4f}, Validation accuracy: {:.4f}".format(avgValLoss, valCorrect))

        if args.checkpointing:
            accelerator.save_state(safe_serialization=False)

        # Update learning rate according to schedule
        scheduler.step(avgValLoss)


    endTime = time.time()
    totalTime = endTime-startTime
    totalTime_clean = time.strftime('%H:%M:%S', time.gmtime(totalTime))
    if accelerator.is_main_process:
        print(f"Total training time: {totalTime_clean}")
        # Save the complete training history
        os.makedirs(results_folder, exist_ok=True)
        np.save(os.path.join(results_folder, "model_history.npy"), history)

    # Save the complete model -> error if CUDA_HOME does not exist
    #accelerator.wait_for_everyone()
    accelerator.save_model(model, results_folder, safe_serialization=False)

    accelerator.end_training()


if __name__ =="__main__":
    worker(args)

