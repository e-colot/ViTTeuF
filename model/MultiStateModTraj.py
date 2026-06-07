import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.PointPillar import PillarSplit,PillarVFE,PillarConv
from model.LSTM_block import CustomLSTM
import numpy as np

class AdvancedTrajectoryPredictor(nn.Module):
    def __init__(self, d_model=128, num_modes=3, future_steps=12):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.vehicle_width = 1.6
        self.pixel_nbr = 32
        self.grid_H = self.pixel_nbr*self.vehicle_width
        self.grid_W = self.pixel_nbr*self.vehicle_width

        # CONTEXT ENCODERS
        resnet = models.resnet18(pretrained=True)
        self.camera_backbone = nn.Sequential(*list(resnet.children())[:-2]) # [B, 512, 7, 7]
        self.camera_projector = nn.Linear(512, d_model)

        # PointPillars layers
        # 1.6m resolution (vehicle width is more or less 1.6m in average)
        # 51.2m for the grid leads to 32x32 pixels grid (51.2/1.6)
        self.splitter = PillarSplit(H=self.grid_H, W=self.grid_W, block_scale=self.vehicle_width, max_voxels=20) 
        self.vfe = PillarVFE(H=self.grid_H, W=self.grid_W, block_scale=self.vehicle_width, layers=[64, 64])
        self.backbone_2d = PillarConv(in_channel=64, out_channel=64, hidden_channels=[64, 128], kernel=3, stride=[1, 2], spatial_compression=1)
        self.lidar_projector = nn.Linear(64, d_model)

        #self.cam_spatial_emb = fun_2DSinusoidal_pos(7 , 7, self.d_model).flatten(0, 1).unsqueeze(0)    # [1, 49, d_model]
        #self.lidar_spatial_emb = fun_2DSinusoidal_pos(self.pixel_nbr, self.pixel_nbr, self.d_model).flatten(0, 1).unsqueeze(0) # [1, 1024, d_model]

        self.register_buffer('cam_spatial_emb', fun_2DSinusoidal_pos(7, 7, self.d_model).flatten(0, 1).unsqueeze(0))
        self.register_buffer('lidar_spatial_emb', fun_2DSinusoidal_pos(self.pixel_nbr, self.pixel_nbr, self.d_model).flatten(0, 1).unsqueeze(0))

        # UPGRADED HISTORY STATE ENCODER
        self.history_encoder = nn.Sequential(
            nn.Linear(2, 64), #In[4, 2] -> Out[4, 64]
            nn.ReLU(),
            nn.Linear(64, d_model) #In[4,64] -> Out[4,128]
        )
        # LSTM - Extract vehicle dynamics [Cell (long term memory) and Hidden state (short term memory)]
        #self.history_lstm = nn.LSTM(d_model, d_model, batch_first=True)

        self.history_lstm = CustomLSTM(d_model, d_model)

        # TRANSFORMER
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # MODE QUERIES (Named as Anchors in Litterature)
        self.mode_queries = nn.Parameter(torch.randn(num_modes, d_model)) # [Batch, 3, 128] Initialize to random 128 values
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        #self.decoder_layers = nn.ModuleList([
        #    nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True),
        #    nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True) 
        #])

        #self.decoder_layer1 = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        # self.decoder_layer2 = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)

        # UPGRADED OUTPUT HEADS
        # Trajectory head outputs X, Y, Heading Angle Theta (3 values per step)
        self.traj_head = nn.Linear(d_model, future_steps * 3) 
        self.prob_head = nn.Linear(d_model, 1)

    def forward(self, image, point_cloud, history):
        """
        image : [Batch, 3, 224, 224]
        point_cloud : [points , 4]
        history : [Batch, 128]
        """
        batch_size = image.size(0)
        device = image.device

        # Context extraction (Camera + Lidar)
        cam_feats = self.camera_backbone(image).permute(0, 2, 3, 1) # [Batch, 512, 7, 7] -> [Batch, 7, 7, 512]
        cam_tokens_2D = cam_feats.flatten(1, 2) #  [Batch, 7, 7, 512] -> [Batch, 49, 512]
        cam_tokens = self.camera_projector(cam_tokens_2D) # [Batch, 49, 512] -> [Batch, 49, d_model]

        # Note: PointPillars processes single point clouds
        # Reminder : Pillars [N_col, N_voxels, 4] | Usage [N_usable_voxels per col] | Idx [X,Y position in the grid 2048x2048]
        pillars, usage, idx = self.splitter(point_cloud)
        # [W_grid, H_grid, 64 features]
        bev_map = self.vfe(pillars, usage, idx)
        # [W_grid, H_grid, 64 features]
        dense_bev = self.backbone_2d(bev_map)
        W, H, C = dense_bev.shape
        # [W_grid * H_grid, 64 features]
        lidar_flat = dense_bev.view(W * H, C)
        # [Batch, W_grid * H_grid, 64 features]
        lidar_tokens = self.lidar_projector(lidar_flat).unsqueeze(0).repeat(batch_size, 1, 1)

        lidar_token_spat = lidar_tokens + self.lidar_spatial_emb
        camera_token_spat = cam_tokens + self.cam_spatial_emb

        # Camera [Batch ,49 , d_model] | Lidar [Batch ,32 * 32, d_model]  => Tot [Batch, 1073, d_model]
        fused_memory = self.transformer_encoder(torch.cat([camera_token_spat, lidar_token_spat], dim=1))

        #  State Query || Associated to vehicle momentum
        # Process history step-by-step to capture physical momentum
        hist_features = self.history_encoder(history) # [Batch, 4, d_model]
        _, (state_query, _) = self.history_lstm(hist_features) 
        state_query = state_query.squeeze(0).unsqueeze(1) # [Batch, 1, d_model]

        # Combine State and Motion queries || Similar to positional embedding addition
        # Duplicate the state query for each mode, and fuse them with our learnable intents
        modes_expanded = self.mode_queries.unsqueeze(0).repeat(batch_size, 1, 1) # [Batch, 3, d_model]
        decoder_queries = state_query.repeat(1, self.num_modes, 1) # [Batch, 3, d_model]

        fused_state = decoder_queries + modes_expanded

        # Decoder Cross-Attention
        #decoder_out_1 = self.decoder_layer1(tgt=fused_state,memory=fused_memory) # [Batch, 3, d_model]
        #fused_state_2 = decoder_out_1 + modes_expanded
        #decoder_out_2 = self.decoder_layer2(tgt=fused_state_2, memory=fused_memory) # [Batch, 3, d_model]
        #decoder_out = decoder_out_2

        decoder_out = self.transformer_decoder(tgt=fused_state,memory=fused_memory) # [Batch, 3, d_model]

        # Reshape to extract [Batch, Mode, Steps, Features(X, Y, Yaw angle)]
        pred_outputs = self.traj_head(decoder_out).view(batch_size, self.num_modes, self.future_steps, 3)
        pred_trajectories = pred_outputs[..., :2]   # [Batch, 3, 12, 2] -> Position prediction | L_ts Targets
        pred_headings = pred_outputs[..., 2]        # [Batch, 3, 12]    -> Angle prediction | L_s Targets
        # pred_velocity = pred_outputs[..., 3]        # [Batch, 3, 12]    -> Velocity prediction | L_s Targets
        
        pred_logits = self.prob_head(decoder_out).squeeze(-1) # [Batch, 3, 1] -> [Batch, 3]

        return pred_trajectories, pred_headings, pred_logits


def fun_2DSinusoidal_pos(H, W, F):

    dim = F // 2
    
    # Meshgrid computation
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    
    # Angular frequency for each feature
    omega = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(np.log(10000.0) / dim))
    
    out_x = torch.zeros(H, W, dim)
    out_y = torch.zeros(H, W, dim)
    
    for i, ang_freq in enumerate(omega):
        out_x[..., 2*i] = torch.sin(x * ang_freq)
        out_x[..., 2*i+1] = torch.cos(x * ang_freq)
        out_y[..., 2*i] = torch.sin(y * ang_freq)
        out_y[..., 2*i+1] = torch.cos(y * ang_freq)
        
    # X context spatial encoding with frequency proper to each feature
    # Concatenation with spatial encoding along Y
    return torch.cat([out_x, out_y], dim=-1)

