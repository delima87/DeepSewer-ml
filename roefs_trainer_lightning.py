import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torchvision import transforms,datasets
import pytorch_lightning as pl
import sewer_models
import ml_models
from torchmetrics.functional import accuracy
import numpy as np

def evaluate(dataloader, model, device):
    model.eval()

    sigmoidPredictions = None
    imgPathsList = []

    sigmoid = nn.Sigmoid()

    dataLen = len(dataloader)
    
    with torch.no_grad():
        for i, (images, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print("{} / {}".format(i, dataLen))

            images = images.to(device)

            output = model(images)            

            sigmoidOutput = sigmoid(output).detach().cpu().numpy()

            if sigmoidPredictions is None:
                sigmoidPredictions = sigmoidOutput
            else:
                sigmoidPredictions = np.vstack((sigmoidPredictions, sigmoidOutput))

            imgPathsList.extend(list(imgPaths))
    return sigmoidPredictions, imgPathsList

class LightningClassifier(pl.LightningModule):

    def __init__(self, backbone, is_binary, pretrained = False):
        super().__init__()
        self.model = backbone
        self.criterion=torch.nn.BCEWithLogitsLoss  
        self.train_function = self.normal_loss  
        self.is_pretrained = pretrained
        self.is_binary = is_binary

    def forward(self, x):
        logits = self.model(x)
        return logits
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)


    def normal_loss(self, x, y):
        y = y.float()
        y_hat = self.model(x)
        loss = self.cross_entropy_loss(y_hat, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        #acc = self.train_accuracy(logits, y)
        loss = self.cross_entropy_loss(logits, y)
        # .log sends to tensorboard/logger, prog_bar also sends to the progress bar
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, sync_dist=True)
        return result

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.cross_entropy_loss(logits, y)
        # lightning monitors 'checkpoint_on' to know when to checkpoint (this is a tensor)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('test_loss', loss, sync_dist=True)
        return result


    def configure_optimizers(self):
        if (self.is_pretrained) & (not self.is_binary):
            optim = torch.optim.SGD(self.model.head.parameters(), lr=0.001, momentum=0.6)
            print("retraining multi")
        elif (self.is_pretrained) & (self.is_binary):
            print("retraining binary")
            optim = torch.optim.SGD(self.model.classifier[-1].parameters(), lr=0.001, momentum=0.6)
        else:
            print("training new model")
            optim = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30, 60, 80], gamma=0.1) 

        return [optim], [scheduler]
    
   

def load_model(model_path):

    model_last_ckpt = torch.load(model_path)
    model_name = model_last_ckpt["hyper_parameters"]["model"]
    num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    
    best_model = model_last_ckpt
    best_model_state_dict = best_model["state_dict"]
    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict, model_name, num_classes

def load_data_train_val(train_path,val_path):
        # Init data with transforms
    img_size = 224

    train_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    eval_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    #loading data
    image_datasets_train = datasets.ImageFolder(train_path,train_transform)
    image_datasets_val = datasets.ImageFolder(val_path,eval_transform)
    train_dl = torch.utils.data.DataLoader(image_datasets_train, batch_size=4, shuffle=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(image_datasets_val, batch_size=4, shuffle=False, num_workers=4)
    dataset_sizes = len(image_datasets_train)
    class_names = image_datasets_train.classes
    print("data loaded size {}, classes {}".format(dataset_sizes,class_names))
    return train_dl, val_dl

def training_tresnet_xl_e2e(model_path,train_path, val_path, out_path, num_epochs,n_new_classes):
    train_dl,val_dl = load_data_train_val(train_path,val_path)
    #load models
    updated_state_dict, model_name, n_classes = load_model(model_path)
    model = ml_models.tresnet_xl(n_classes)
    model.load_state_dict(updated_state_dict)
    print("loaded model {} number of classes {}".format(model_name,n_classes))
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = 2656 
    model.head = nn.Linear(num_ftrs,n_new_classes)
    modelLit = LightningClassifier(model,is_binary=False,pretrained=True)
    trainer = pl.Trainer(max_epochs=num_epochs,gpus=1)
    trainer.fit(modelLit,train_dl,val_dl)
    torch.save({
        'state_dict': modelLit.state_dict(),
        'name': model_name,
        'num_classes': n_classes
        }, out_path)
    print("new output layer", model.head)
    print("Done")
    

def training_binary_xie2019(model_path,train_path, val_path, out_path, num_epochs,n_new_classes):
    pl.seed_everything(1234567890)
    train_dl,val_dl = load_data_train_val(train_path,val_path)
    #load models
    updated_state_dict, model_name, n_classes = load_model(model_path)
    model = sewer_models.__dict__[model_name](num_classes = n_classes)
    model.load_state_dict(updated_state_dict)
   
    print("loaded model {} number of classes {}".format(model_name,n_classes))
    # #training with features
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = 512 
    model.classifier[-1] = nn.Linear(num_ftrs,n_new_classes)
    modelLit = LightningClassifier(model,is_binary=True,pretrained=True)
    trainer = pl.Trainer(max_epochs=num_epochs,gpus=1)
    trainer.fit(modelLit,train_dl,val_dl)
    
     
    torch.save({
            'state_dict': modelLit.state_dict(),
            'name': model_name,
            'num_classes': num_classes
            }, out_path)

    


if __name__ == "__main__":
    model_path = "models/xie2019_binary-binary-version_1.pth"    
    train_path = "roefs_one_classs/train"    
    val_path = "roefs_one_classs/val"
    out_path = "checkpoints/retrained.pth"  
    num_epochs = 2 
    num_classes = 1   
    training_binary_xie2019(model_path,train_path,val_path, out_path, num_epochs,num_classes)