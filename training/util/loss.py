# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 


import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
class InstanceSegmentationLoss(nn.Module):
    """
    Custom Segmentation Loss combining:
      - Intra-instance variance (Huber loss toward the mean for each instance).
      - Inter-instance separation (penalizing closeness of pixels from other instances).
      - Mean-level loss (repulsion among means of different instances).

    Excludes background-labeled pixels (0, 0, 0) unless no_bg[batch] is False.

    Args:
        None    

    Inputs:
        prediction (torch.Tensor): [B, 3, H, W] in [0, 255].
        target (torch.Tensor):     [B, 3, H, W] in [0, 255].
        no_bg (torch.Tensor):      Boolean array of shape [B], where True indicates that
                                   background should not be included for that batch.

    Outputs:
        torch.Tensor: Scalar loss.
    """

    def __init__(self):
        super(InstanceSegmentationLoss, self).__init__()
        self.name = "InstanceSegmentationLoss"

    def forward(self, prediction, target, no_bg):
        """
        Forward pass to compute the custom segmentation loss.
        """
        # Ensure predictions and targets are float tensors
        prediction = prediction.float()
        target     = target.float()

        batch_size, channels, height, width = prediction.shape
        total_loss = 0.0

        # We iterate over the batch dimension explicitly
        for batch_idx in range(batch_size):
            loss = 0.0
            ct   = 0   # count how many instance-related terms contributed

            # Permute predicted channels to [H, W, 3] for easier indexing
            pred_i = prediction[batch_idx].permute(1, 2, 0)  # [H, W, 3]
            gt_i   = target[batch_idx]                       # [3, H, W]
            gt_i_permute = gt_i.permute(1, 2, 0)             # [H, W, 3]

            # Flatten ground-truth instance map to get unique colors (instances)
            gt_i_flat = gt_i_permute.reshape(-1, 3)          # [H*W, 3]
            unique_instances = torch.unique(gt_i_flat, dim=0)# [num_unique, 3]

            # Keep track of means for the non-background instances
            instance_means = []

            # ---------- Main loop over unique instances ----------
            for inst_id in unique_instances:
                if ct > 1250:
                    continue
                # Create a boolean mask for the current instance
                instance_mask = (
                    (gt_i[0] == inst_id[0]) &
                    (gt_i[1] == inst_id[1]) &
                    (gt_i[2] == inst_id[2])
                )

                # Extract the predicted values for pixels belonging to this instance
                instance_pred = pred_i[instance_mask]  # shape: [num_pixels_in_instance, 3]
                if instance_pred.numel() == 0:
                    continue  # Skip if no pixels belong to this instance

                # Compute the mean prediction for this instance
                mean_inst = instance_pred.mean(dim=0)  # shape: [3]
                
                # Check if this is background (sum of inst_id near 0)
                is_background = (torch.sum(inst_id).abs() < 1e-5)

                # ------ Handle background ------
                # Virtual Kitti 2 is not fully annotated (only cars are labeled, so the "background" can contain objects such as trees or traffic lights)
                # Thus, we don't compute loss on the background mask if the sample comes from Virtual Kitti 2
                if is_background:
                    # If background is NOT to be ignored for this batch
                    if not no_bg[batch_idx]:
                        # Force background pixels to be near (0,0,0)
                        var_loss = nn.functional.huber_loss(
                            instance_pred,
                            torch.zeros_like(instance_pred)
                        )
                        loss += var_loss
                        ct   += 1
                        instance_means.append(torch.tensor([0, 0, 0]).cuda())
                    else:
                        # If background is ignored, skip
                        continue
                else:
                    # Intra-instance variance: push instance_pred toward mean_inst
                    var_loss = nn.functional.huber_loss(
                        instance_pred,
                        mean_inst.unsqueeze(0).expand_as(instance_pred)
                    )
                    loss += var_loss
                    ct   += 1

                    # Keep track of the mean for further mean-level separation
                    instance_means.append(mean_inst)

                # ------ Inter-instance separation from other pixels ------
                non_instance_pred = pred_i[~instance_mask]  # shape: [num_pixels_not_in_instance, 3]
                if non_instance_pred.numel() > 0 and not is_background:
                    size  = instance_mask.sum()  # number of pixels in this instance
                    w     = 10.0 / torch.sqrt(size.float()) #ORIGINALLY 30 FOR SD2
                    lambda_sep = 300
                    # squared L2 distance from each non-instance pixel to this instance's mean
                    distances   = (non_instance_pred - mean_inst).pow(2).sum(dim=1)
                    separation  = torch.mean(lambda_sep / (1.0 + distances))
                    loss       += w * separation

            # -------------- Vectorized Mean-Level Loss ---------------
            # We now have a list of means for each non-background instance: instance_means
            # Let's push them away from each other if they are too close.
            if len(instance_means) > 1:
                means = torch.stack(instance_means, dim=0)[:ct, :]  # shape: [num_means, 3]
                # Compute the pairwise squared distances in a vectorized manner:
                # differences: [num_means, num_means, 3]
                differences = means.unsqueeze(1) - means.unsqueeze(0)
                # squared_distances: [num_means, num_means]
                squared_distances = differences.pow(2).sum(dim=2)

                # We only want i < j to avoid double-counting or i=j
                # shape: [2, #pairs]
                i_indices, j_indices = torch.triu_indices(
                    squared_distances.size(0),
                    squared_distances.size(1),
                    offset=1
                )
                pairwise_dists = squared_distances[i_indices, j_indices]

                # Simple reciprocal penalty or any function of distance
                # penalty[i,j] = alpha / (dist + eps)
                lambda_mean   = 300.0
                eps     = 1
                penalty = lambda_mean / (pairwise_dists + eps)

                # Average across all pairs
                mean_separation_loss = penalty.mean()
                loss += mean_separation_loss

            # Avoid dividing by zero if ct was never incremented
            if ct == 0:
                ct = 1

            total_loss += loss / float(ct)

        # Average across batch
        return total_loss / float(batch_size)