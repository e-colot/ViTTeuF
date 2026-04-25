from utils import load_utils

PATH_TO_DATA = './data' 

train_loader = load_utils.get_dataloader(PATH_TO_DATA, batch_size=8)

for batch_idx, (images, tokens) in enumerate(train_loader):
    print(f"Batch {batch_idx}: {images.shape}")
    
    # Your model logic here:
    # output = model(images)
    
    if batch_idx == 2: # Just a quick test
        break

