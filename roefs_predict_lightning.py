import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from torchvision import transforms,datasets
import pytorch_lightning as pl
import sewer_models
from roefs_trainer_lightning import LightningClassifier
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



def load_model(model_path):

    model_last_ckpt = torch.load(model_path)
    model_name = model_last_ckpt["name"]
    num_classes = model_last_ckpt["num_classes"]
    # num_classes = model_last_ckpt["hyper_parameters"]["num_classes"]
    # training_mode = model_last_ckpt["hyper_parameters"]["training_mode"]
    # br_defect = model_last_ckpt["hyper_parameters"]["br_defect"]
    
    # best_model = model_last_ckpt
    best_model_state_dict = model_last_ckpt["state_dict"]
    updated_state_dict = OrderedDict()
    for k,v in best_model_state_dict.items():
        name = k.replace("model.", "")
        if "criterion" in name:
            continue

        updated_state_dict[name] = v

    return updated_state_dict, model_name, num_classes



def main():
    pl.seed_everything(1234567890)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init data with transforms
    img_size = 224


    eval_transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #loading data
    image_datasets_test = datasets.ImageFolder("roefs_one_classs/val",eval_transform)
    test_dl = torch.utils.data.DataLoader(image_datasets_test, batch_size=4, shuffle=True, num_workers=4)
    #load model
    MODEL_PATH = "models/rerained.pth"
    # Validation results
    updated_state_dict, model_name, num_classes = load_model(MODEL_PATH)
    model_trained = sewer_models.Xie2019(1)
    model_trained.load_state_dict(updated_state_dict,strict=False)
    model_trained = model_trained.to(device)
    print("VALIDATION")
    sigmoid_predictions, val_imgPaths = evaluate(test_dl, model_trained, device)
    print(sigmoid_predictions) 



if __name__ == "__main__":
    main()