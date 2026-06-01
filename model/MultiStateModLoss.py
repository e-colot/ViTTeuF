import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalTrajectoryLoss(nn.Module):
    def __init__(self, cls_weight=1.0, state_weight=2.0, num_mode = 3):
        super().__init__()
        self.cls_weight = cls_weight
        self.state_weight = state_weight
        self.reg_criterion = nn.SmoothL1Loss(reduction='none')
        self.num_mode = num_mode

    def forward(self, pred_trajs, pred_headings, pred_logits, gt_traj):
        """
        Inputs:
            pred_trajs:    [Batch, 3, 12, 2]  --> Predicted X, Y paths
            pred_headings: [Batch, 3, 12]     --> Predicted heading angles
            pred_logits:   [Batch, 3]         --> Intent confidence scores
            gt_traj:       [Batch, 12, 2]     --> Ground truth X,Y paths
        """
        batch_size = pred_trajs.size(0)
        
        # Calculate ground truth heading angles on the fly using delta positions (Physics State)
        # Heading theta = arctan2(delta_y, delta_x)
        delta_coords = gt_traj[:, 1:, :] - gt_traj[:, :-1, :] # Delta X,Y [Batch, 11, 2]
        gt_headings = torch.atan2(delta_coords[..., 1], delta_coords[..., 0]) #Yaw angle [Batch, 11]
        # Duplicate first elem to keep dimension [Batch, 12]
        gt_headings = torch.cat([gt_headings[:, :1], gt_headings], dim=1).unsqueeze(1) # [Batch, 1, 12]

        # ---------------------------------------------------------
        # 1. FIND THE WINNER USING SPATIAL TRAJECTORY
        # ---------------------------------------------------------
        gt_expanded = gt_traj.unsqueeze(1)
        gt_expanded = gt_expanded.repeat(1,self.num_mode,1,1)
        traj_errors = self.reg_criterion(pred_trajs, gt_expanded).sum(dim=[-2, -1]) # [Batch, 3]
        winning_errors, winning_indices = torch.min(traj_errors, dim=1)

        # Slice the batches and select the winning mode from the prediction !
        # Operation like pred_traj[:, winning_indices] is not allowed !
        batch_idx = torch.arange(batch_size, device=pred_trajs.device)
        best_trajs = pred_trajs[batch_idx, winning_indices]     # [Batch, 12, 2]
        best_headings = pred_headings[batch_idx, winning_indices] # [Batch, 12]

        # ---------------------------------------------------------
        # 2. COMPUTE L_ts (Trajectory Sequence Loss with Time Weights)
        # ---------------------------------------------------------
        # Create sequential time weights: steps further out in the future cost more!
        # Compound system for errors in time | missing a traj at the final frame holds more significancy than at the first frame 
        time_steps = torch.arange(1, 13, device=pred_trajs.device).float() # tensor[1,2,..,12]
        time_weights = torch.exp(time_steps / 12.0).view(1, 12, 1) # [exp(1/12) ~ 1.08, ..., exp(12/12) ~ 2.71]

        raw_ts_loss = self.reg_criterion(best_trajs, gt_traj) # Raw_ts [Batch, 12, 2]
        loss_ts = (raw_ts_loss * time_weights).mean() # mean(Raw_ts .* weights) = [1]

        # ---------------------------------------------------------
        # 3. COMPUTE L_s (State / Static Heading Alignment Loss)
        # ---------------------------------------------------------
        # Compares predicted heading angles against reality
        loss_s = self.reg_criterion(best_headings, gt_headings.squeeze(1)).mean() # mean(best_Yaw_angle - GT_Yaw_angle) -> [1]

        # ---------------------------------------------------------
        # 4. COMPUTE CLASSIFICATION
        # ---------------------------------------------------------
        loss_cls = F.cross_entropy(pred_logits, winning_indices)

        # Combine into Multi-Task Formula
        total_loss = loss_ts + (self.state_weight * loss_s) + (self.cls_weight * loss_cls)

        return total_loss, loss_ts, loss_s, loss_cls