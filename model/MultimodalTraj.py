import torch
import torch.nn as nn
import torchvision.models as models
from model.PointPillar import PillarSplit,PillarVFE,PillarConv

class MultiModalTrajectoryPredictor(nn.Module):
    def __init__(self, d_model=128, num_modes=3, future_steps=12):
        super().__init__()
        self.d_model = d_model
        self.num_modes = num_modes
        self.future_steps = future_steps

        # ---------------------------------------------------------
        # 1. MODALITY TOKENIZERS
        # ---------------------------------------------------------
        # A. Camera Branch (ResNet-18)
        resnet = models.resnet18(pretrained=True)
        self.camera_backbone = nn.Sequential(*list(resnet.children())[:-2]) # Stops at [B, 512, 7, 7]
        self.camera_projector = nn.Linear(512, d_model)
        
        ### CUSTOM RESNET ###
        # Not suitable as the weights are not pre-trained and result to a slower convergence of the loss
        #self.camera_backbone = CustomResNet18Backbone()

        # B. LiDAR Branch (Your PointPillars Blocks)
        # Assuming H, W setup creates a 32x32 BEV grid
        self.splitter = PillarSplit(H=51.2, W=51.2, block_scale=1.6, max_voxels=20) 
        self.vfe = PillarVFE(H=51.2, W=51.2, block_scale=1.6, layers=[64, 64])
        self.backbone_2d = PillarConv(in_channel=64, out_channel=64, hidden_channels=[64, 128], kernel=3, stride=[1, 2], spatial_compression=1)
        self.lidar_projector = nn.Linear(64, d_model)

        # C. History Trajectory Branch
        self.history_encoder = nn.Sequential(
            nn.Linear(4 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        # ---------------------------------------------------------
        # 2. TRANSFORMER CORE
        # ---------------------------------------------------------
        # Encoder fuses Camera + LiDAR + History tokens together
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Learnable Intent Queries
        self.mode_queries = nn.Parameter(torch.randn(num_modes, d_model))
        
        # Decoder generates paths using the fused memory
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # ---------------------------------------------------------
        # 3. OUTPUT HEADS
        # ---------------------------------------------------------
        self.traj_head = nn.Linear(d_model, future_steps * 2)
        self.prob_head = nn.Linear(d_model, 1)

    def forward(self, image, point_cloud, history):
        """
        Args:
            Reminder shape : [Batch, Features, Height, Width]
            image: Camera tensor [Batch, 3, 224, 224]
            point_cloud: Raw unbatched LiDAR point cloud [L, 4] (processed frame-by-frame)
            history: History coordinates [Batch, 4, 2]
        """
        batch_size = image.size(0)

        # --- Step 1: Tokenize Camera ---
        cam_feats = self.camera_backbone(image)                  # Input : [Batch, 3, 224, 224] -> [Batch, 512, 7, 7]
        cam_feats = cam_feats.permute(0, 2, 3, 1).flatten(1, 2)  # [Batch, 49, 512]
        cam_tokens = self.camera_projector(cam_feats)            # [Batch, 49, 128]

        # --- Step 2: Tokenize LiDAR ---
        # Note: PointPillars processes single point clouds
        # Reminder : Pillars [N_col, N_voxels, 4] | Usage [N_usable_voxels per col] | Idx [X,Y position in the grid 2048x2048]
        pillars, usage, idx = self.splitter(point_cloud)
        bev_map = self.vfe(pillars, usage, idx)                  # [W, H, 64]
        dense_bev = self.backbone_2d(bev_map)                    # [W, H, 64]
        
        W, H, C = dense_bev.shape
        lidar_flat = dense_bev.view(W * H, C)                    # [W*H, 64]
        lidar_tokens = self.lidar_projector(lidar_flat)          # [W*H, 128]
        lidar_tokens = lidar_tokens.unsqueeze(0).repeat(batch_size, 1, 1) # [Batch, 1024, 128]

        # --- Step 3: Tokenize History ---
        hist_flat = history.view(batch_size, -1)                 # [Batch, 8]
        hist_tokens = self.history_encoder(hist_flat).unsqueeze(1) # [Batch, 1, 128]

        # --- Step 4: Multi-Modal Concatenation ---
        # Sequence total length = 49 (Cam) + 1024 (LiDAR) + 1 (Hist) = 1074 tokens
        # Target Shape: [Batch, 1074, 128]
        combined_memory = torch.cat([cam_tokens, lidar_tokens, hist_tokens], dim=1)

        # --- Step 5: Encoder Fusion ---
        # Tensors interact across modalities globally
        fused_memory = self.transformer_encoder(combined_memory) # [Batch, 1074, 128]

        # --- Step 6: Decoder Query Extraction ---
        queries = self.mode_queries.unsqueeze(0).repeat(batch_size, 1, 1) # [Batch, K, 128]
        decoder_out = self.transformer_decoder(tgt=queries, memory=fused_memory) # [Batch, K, 128]

        # --- Step 7: Output Generation ---
        pred_trajectories = self.traj_head(decoder_out).view(batch_size, self.num_modes, self.future_steps, 2)
        pred_logits = self.prob_head(decoder_out).squeeze(-1)    # [Batch, K]

        return pred_trajectories, pred_logits