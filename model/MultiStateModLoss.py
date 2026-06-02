import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalTrajectoryLoss(nn.Module):
    def __init__(self, cls_weight=2.0, state_weight=2.0, num_mode = 3, variance = 2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.state_weight = state_weight
        self.reg_criterion = nn.SmoothL1Loss(reduction='none')
        self.num_mode = num_mode
        self.variance = variance

    def forward(self, pred_trajs, pred_headings, pred_logits, gt_traj):
        """
        Inputs:
            pred_trajs:    [Batch, 3, 12, 2]  --> Predicted X, Y paths
            pred_headings: [Batch, 3, 12]     --> Predicted heading angles
            pred_velocity: [Batch, 3, 12]     --> Predicted velocity
            pred_logits:   [Batch, 3]         --> Intent confidence scores
            gt_traj:       [Batch, 12, 2]     --> Ground truth X,Y paths
        """
        batch_size = pred_trajs.size(0)
        
        # Calculate ground truth heading angles on the fly using delta positions
        # Yaw angle = arctan2(delta_y, delta_x)
        delta_coords = gt_traj[:, 1:, :] - gt_traj[:, :-1, :] # Delta X,Y [Batch, 11, 2]
        gt_headings = torch.atan2(delta_coords[..., 1], delta_coords[..., 0]) # Yaw angle [Batch, 11]
        # Velocity state || Not used - taking features ressources without noticeable progress
        gt_velocity = torch.norm(delta_coords,p=2,dim=-1)
        gt_velocity = gt_velocity/0.5
        # Duplicate first elem to keep dimension [Batch, 12]
        gt_headings = torch.cat([gt_headings[:, :1], gt_headings], dim=1).unsqueeze(1) # [Batch, 1, 12]
        gt_velocity = torch.cat([gt_velocity[:, :1],gt_velocity], dim = 1).unsqueeze(1) # [Batch, 1, 12]

        # FINDING WINNER INDEX
        gt_expanded = gt_traj.unsqueeze(1)
        gt_expanded = gt_expanded.repeat(1,self.num_mode,1,1)

        disp_errors = torch.norm(pred_trajs - gt_expanded, p=2, dim=-1) # [Batch, K, 12]
        traj_errors_sum = disp_errors.sum(dim=-1) # [Batch, K]
        winning_errors, winning_indices = torch.min(traj_errors_sum, dim=1) # get the winning index 

        # Slice the batches and select the winning mode from the prediction !
        # Operation like pred_traj[:, winning_indices] is not allowed !
        batch_idx = torch.arange(batch_size, device=pred_trajs.device)
        best_trajs = pred_trajs[batch_idx, winning_indices]     # [Batch, 12, 2]
        best_headings = pred_headings[batch_idx, winning_indices] # [Batch, 12]
        # best_velocity = pred_velocity[batch_idx,winning_indices]  # [Batch, 12]

        # COMPUTE L_ts [Trajectory Sequence - L1 Regression with step weight]
        # Create sequential time weights: steps further out in the future cost more!
        # Compound system for errors in time | missing a traj at the final frame holds more significancy than at the first frame 
        time_steps = torch.arange(1, 13, device=pred_trajs.device).float() # tensor[1,2,..,12]
        time_weights = torch.exp(time_steps / 12.0).view(1, 12, 1) # [exp(1/12) ~ 1.08, ..., exp(12/12) ~ 2.71]

        raw_ts_loss = self.reg_criterion(best_trajs, gt_traj) # Raw_ts [Batch, 12, 2]
        loss_ts = (raw_ts_loss * time_weights).mean() # mean(Raw_ts .* weights) = [1]

        # COMPUTE L_s (Cosine difference)
        # Compares predicted yaw angles against reality
        loss_heading = (1.0 - torch.cos(best_headings - gt_headings)).mean() # cos avoid errors due to corssing the limit -pi/pi -> [1]
        loss_s = loss_heading
        # loss_velocity = self.reg_criterion(best_velocity, gt_velocity.squeeze(1)).mean() # mean(best_vel - GT_vel) -> [1]
        # loss_s = loss_heading + loss_velocity


        # ---------------------------------------------------------
        # COMPUTE L_cls (Classification error - Soft target [Gradient descent over all modes])
        # Winner-takes-all strategy gives bouncing values for L_cls
        FDE_per_batch = disp_errors[...,-1] # [Batch, K]
        soft_target = torch.exp(-FDE_per_batch/self.variance)
        # Avoid harsh penality in case of winning all index selection
        # Reminder : -ln(exp(elem)/sum(exp(elem)))
        loss_cls = F.binary_cross_entropy_with_logits(pred_logits, soft_target)

        #loss_cls = F.cross_entropy(pred_logits, winning_indices) 

        # COMBINE LOSSES
        total_loss = loss_ts + (self.state_weight * loss_s) + (self.cls_weight * loss_cls)

        return total_loss, loss_ts, loss_s, loss_cls