import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalTrajectoryLoss(nn.Module):
    def __init__(self, cls_weight=1.0):
        super().__init__()
        self.cls_weight = cls_weight
        # Smooth L1 is great for trajectories because it handles outliers gracefully
        self.reg_criterion = nn.SmoothL1Loss(reduction='none') 

    def forward(self, pred_trajs, pred_logits, gt_traj):
        """
        Inputs:
            pred_trajs:  [Batch, K, 12, 2]  --> The 3 predicted paths
            pred_logits: [Batch, K]         --> The 3 raw confidence scores (logits)
            gt_traj:     [Batch, 12, 2]     --> The 1 real path the car actually took
        """
        batch_size = pred_trajs.size(0)
        num_modes = pred_trajs.size(1)

        # ---------------------------------------------------------
        # STEP 1: FIND THE WINNER (Closest Path)
        # ---------------------------------------------------------
        # Expand ground truth shape from [B, 12, 2] to [B, 1, 12, 2] so we can subtract it
        gt_expanded = gt_traj.unsqueeze(1)
        
        # Calculate the error for every step across all modes
        # Shape results in: [Batch, K, 12, 2]
        reg_loss_all = self.reg_criterion(pred_trajs, gt_expanded)
        
        # Sum the errors per path to get an overall score for each mode
        # Shape results in: [Batch, K]
        mode_errors = reg_loss_all.sum(dim=[-2, -1]) 
        
        # Find the index (0, 1, or 2) of the path with the MINIMUM total error
        # winning_indices shape: [Batch]
        winning_errors, winning_indices = torch.min(mode_errors, dim=1)

        # ---------------------------------------------------------
        # STEP 2: COMPUTE REGRESSION LOSS (Only for the Winner)
        # ---------------------------------------------------------
        # We average the displacement error of ONLY the closest path
        loss_reg = winning_errors.mean()

        # ---------------------------------------------------------
        # STEP 3: COMPUTE CLASSIFICATION LOSS (Tricking the Probability Head)
        # ---------------------------------------------------------
        # The target label for our probability head is the index of the winning path!
        # We want the network to give 100% confidence to the path that performed the best.
        loss_cls = F.cross_entropy(pred_logits, winning_indices)

        # ---------------------------------------------------------
        # STEP 4: TOTAL FUSED LOSS
        # ---------------------------------------------------------
        total_loss = loss_reg + (self.cls_weight * loss_cls)

        return total_loss, loss_reg, loss_cls