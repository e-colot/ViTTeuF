from utils import load_utils

PATH_TO_DATA = './data' 
BATCH_SIZE = 1

train_loader = load_utils.get_dataloader(dataroot=PATH_TO_DATA, batch_size=BATCH_SIZE)

for points, anns, tokens in train_loader:
    
    # points is a list of tensors of shape:
    # [B](N, 4)
    # this notation means a python list of 8 elements, each element being a tensor of Nx4 elements
    # B is the batch size, defined in the train_loader definition
    # Because the N (number of lidar points) is varying, it cannot be in a 3D tensor

    # anns (for annotations) is a 2D list of dictionaries
    # [B, N_el]
    # B is the batch size and N_el the number of elements in the scene
    # The dictionary of each element contains:
    #   'center': numpy array of 3 elements, (x, y, z)
    #   'size': numpy array of 3 elements, (x, y, z)
    #   'name': string
    #   'token': string

    # tokens, a string identifying the frame if no misinterpretation happenend

    print('This was the test')
    break

