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
    
class BinarySegmentationLoss(nn.Module):
    def __init__(self, channels=3):
        super(BinarySegmentationLoss, self).__init__()
        self.name = "BinarySegmentationLoss"
        self._whiteLoss = 0.0
        self._blackLoss = 0.0
        self._separationLoss = 0.0
        
    @property
    def whiteLoss(self):
        _ = self._whiteLoss
        self._whiteLoss = 0.0  # Reset after accessing
        return _

    @property
    def blackLoss(self):
        _ = self._blackLoss
        self._blackLoss = 0.0  # Reset after accessing
        return _

    @property
    def separationLoss(self):
        _ = self._separationLoss
        self._separationLoss = 0.0  # Reset after accessing
        return _
    
    def forward(self, prediction, target):
        """
        Binary segmentation loss for document forgery detection.
        
        Args:
            if channels == 1:
            prediction (torch.Tensor): [B, 1, H, W] in [0, 255] (model output)
            target (torch.Tensor):     [B, 1, H, W] in [0, 255] (authentic=0, forgery=255)
            
            if channels == 3:
            prediction (torch.Tensor): [B, 3, H, W] in [0, 255] (model output)
            target (torch.Tensor):     [B, 3, H, W] in [0, 255] (authentic=0, forgery=255 in all channel)
            
        Returns:
            torch.Tensor: Scalar loss
        """
        prediction = prediction.float()
        target = target.float()
        batch_size, _, height, width = prediction.shape
        total_loss = 0.0
        
        # Define class colors - use same device as input tensors
        device = prediction.device
        if prediction.shape[1] == 1:  # Single channel case
            AUTHENTIC_COLOR = torch.tensor([0.0], device=device)
            FORGERY_COLOR = torch.tensor([255.0], device=device)
        else:
            AUTHENTIC_COLOR = torch.tensor([0.0, 0.0, 0.0], device=device)
            FORGERY_COLOR = torch.tensor([255.0, 255.0, 255.0], device=device)
        
        for batch_idx in range(batch_size):
            # Permute to [H, W, 3]
            # print(f"prediction[batch_idx].shape: {prediction[batch_idx].shape}")
            # print(f"target[batch_idx].shape: {target[batch_idx].shape}")
            pred = prediction[batch_idx].permute(1, 2, 0)  # [H, W, 3]
            tgt = target[batch_idx].permute(1, 2, 0)       # [H, W, 3]
            
            # Debug: print ranges of pred and tgt
            # print(f"pred range: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
            # print(f"tgt range: [{tgt.min().item():.2f}, {tgt.max().item():.2f}]")
            
            # Create masks - fix forgery mask to match FORGERY_COLOR
            bg_mask = (tgt == AUTHENTIC_COLOR).all(dim=-1)  # Authentic regions
            fg_mask = (tgt == FORGERY_COLOR).all(dim=-1)    # Forged regions (white)
            
            loss = 0.0
            valid_losses = 0
            
            # 1. Authentic region loss - push toward black
            if bg_mask.any():
                bg_pred = pred[bg_mask]
                # print(f"bg_pred.shape: {bg_pred.shape}")
                loss_bg = F.huber_loss(
                    bg_pred, 
                    torch.zeros_like(bg_pred), 
                    reduction='mean'
                )
                loss += loss_bg
                self._blackLoss += loss_bg.item()
                valid_losses += 1
                
            # 2. Forgery region loss - push toward white
            if fg_mask.any():
                fg_pred = pred[fg_mask]
                forgery_target = FORGERY_COLOR.expand_as(fg_pred)
                loss_fg = F.huber_loss(
                    fg_pred, 
                    forgery_target,
                    reduction='mean'
                )
                loss += loss_fg
                self._whiteLoss += loss_fg.item()
                valid_losses += 1
                
            # 3. Class separation loss (critical for subtle forgeries)
            if bg_mask.any() and fg_mask.any():
                mean_authentic = pred[bg_mask].mean(dim=0)
                mean_forgery = pred[fg_mask].mean(dim=0)
                
                # Squared Euclidean distance between class means
                mean_distance = torch.sum((mean_authentic - mean_forgery) ** 2)
                
                # Maximize separation (minimize 1/distance)
                # Using 1/(1+distance) to avoid exploding gradients
                separation_loss = 300.0 / (1.0 + mean_distance)
                loss += separation_loss
                self._separationLoss += separation_loss.item()
                valid_losses += 1
            
            # Normalize by number of active loss terms
            if valid_losses > 0:
                total_loss += loss / valid_losses
        
        # Return average loss across batch
        return total_loss / batch_size


class BinarySegmentationLossV2(nn.Module):
    def __init__(self):
        super(BinarySegmentationLossV2, self).__init__()
        self.name = "BinarySegmentationLossV2"
        self.BCE = nn.BCELoss(reduction='sum')  # Use BCELoss instead of BCEWithLogitsLoss
        self._whiteLoss = 0.0
        self._blackLoss = 0.0
        self._separationLoss = 0.0
        self._ceLoss = 0.0
        
    @property
    def whiteLoss(self):
        _ = self._whiteLoss
        self._whiteLoss = 0.0  # Reset after accessing
        return _

    @property
    def blackLoss(self):
        _ = self._blackLoss
        self._blackLoss = 0.0  # Reset after accessing
        return _

    @property
    def separationLoss(self):
        _ = self._separationLoss
        self._separationLoss = 0.0  # Reset after accessing
        return _
    
    @property
    def ceLoss(self):
        _ = self._ceLoss
        self._ceLoss = 0.0  # Reset after accessing
        return _
    
    def forward(self, prediction, target):
        """
        Binary segmentation loss for document forgery detection.
        
        Args:
            prediction (torch.Tensor): [B, 1, H, W] in [0, 255] (model output)
            target (torch.Tensor):     [B, 1, H, W] in [0, 255] (authentic=0, forgery=255)

        Returns:
            torch.Tensor: Scalar loss
        """
        prediction = prediction.float()
        target = target.float()
        batch_size, _, height, width = prediction.shape
        total_loss = 0.0
        
        # Define class colors - use same device as input tensors
        device = prediction.device
        AUTHENTIC_COLOR = torch.tensor([0.0], device=device)
        FORGERY_COLOR = torch.tensor([255.0], device=device)
        
        for batch_idx in range(batch_size):
            # Permute to [H, W, 3]
            # print(f"prediction[batch_idx].shape: {prediction[batch_idx].shape}")
            # print(f"target[batch_idx].shape: {target[batch_idx].shape}")
            pred = prediction[batch_idx].permute(1, 2, 0)  # [H, W, 3]
            tgt = target[batch_idx].permute(1, 2, 0)       # [H, W, 3]
            
            # Debug: print ranges of pred and tgt
            # print(f"pred range: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
            # print(f"tgt range: [{tgt.min().item():.2f}, {tgt.max().item():.2f}]")
            
            # Create masks - fix forgery mask to match FORGERY_COLOR
            bg_mask = (tgt == AUTHENTIC_COLOR).all(dim=-1)  # Authentic regions
            fg_mask = (tgt == FORGERY_COLOR).all(dim=-1)    # Forged regions (white)
            
            loss = 0.0
            valid_losses = 0
            
            # 1. Authentic region loss - push toward black
            if bg_mask.any():
                bg_pred = pred[bg_mask]
                # print(f"bg_pred.shape: {bg_pred.shape}")
                loss_bg = F.huber_loss(
                    bg_pred, 
                    torch.zeros_like(bg_pred), 
                    reduction='mean'
                )
                loss += loss_bg
                self._blackLoss += loss_bg.item()
                valid_losses += 1
                
            # 2. Forgery region loss - push toward white
            if fg_mask.any():
                fg_pred = pred[fg_mask]
                forgery_target = FORGERY_COLOR.expand_as(fg_pred)
                loss_fg = F.huber_loss(
                    fg_pred, 
                    forgery_target,
                    reduction='mean'
                )
                loss += loss_fg
                self._whiteLoss += loss_fg.item()
                valid_losses += 1
                
            # 3. Class separation loss (critical for subtle forgeries)
            if bg_mask.any() and fg_mask.any():
                mean_authentic = pred[bg_mask].mean(dim=0)
                mean_forgery = pred[fg_mask].mean(dim=0)
                
                # Squared Euclidean distance between class means
                mean_distance = torch.sum((mean_authentic - mean_forgery) ** 2)
                
                # Maximize separation (minimize 1/distance)
                # Using 1/(1+distance) to avoid exploding gradients
                separation_loss = 300.0 / (1.0 + mean_distance)
                loss += separation_loss
                self._separationLoss += separation_loss.item()
                valid_losses += 1
            
            # Normalize by number of active loss terms
            if valid_losses > 0:
                total_loss += loss / valid_losses
        
        # Normalise
        # Normalize target and prediction from [0, 255] to [0, 1]
        target_normalized = target / 255.0
        prediction_normalized = prediction / 255.0
        bce_loss = self.BCE(prediction_normalized, target_normalized)
        self._ceLoss += bce_loss.item()
        total_loss += bce_loss
        
        # Return average loss across batch
        return total_loss / batch_size


class BinarySegmentationLossV3(nn.Module):
    def __init__(self):
        super(BinarySegmentationLossV3, self).__init__()
        self.name = "BinarySegmentationLossV3"
        self._whiteLoss = 0.0
        self._blackLoss = 0.0
        self._separationLoss = 0.0
        
    @property
    def whiteLoss(self):
        _ = self._whiteLoss
        self._whiteLoss = 0.0  # Reset after accessing
        return _

    @property
    def blackLoss(self):
        _ = self._blackLoss
        self._blackLoss = 0.0  # Reset after accessing
        return _

    @property
    def separationLoss(self):
        _ = self._separationLoss
        self._separationLoss = 0.0  # Reset after accessing
        return _
    
    def forward(self, prediction, target):
        """
        Binary segmentation loss for document forgery detection.
        
        Args:
            prediction (torch.Tensor): [B, 3, H, W] in [0, 255] (model output)
            target (torch.Tensor):     [B, 3, H, W] in [0, 255] (authentic=0, forgery=255 in all channel)
            
        Returns:
            torch.Tensor: Scalar loss
        """
        prediction = prediction.float()
        target = target.float()
        batch_size, _, height, width = prediction.shape
        total_loss = 0.0
        
        # Define class colors - use same device as input tensors
        device = prediction.device
        if prediction.shape[1] == 1:  # Single channel case
            AUTHENTIC_COLOR = torch.tensor([0.0], device=device)
            FORGERY_COLOR = torch.tensor([255.0], device=device)
        else:
            AUTHENTIC_COLOR = torch.tensor([0.0, 0.0, 0.0], device=device)
            FORGERY_COLOR = torch.tensor([255.0, 255.0, 255.0], device=device)
        
        for batch_idx in range(batch_size):
            # Permute to [H, W, 3]
            # print(f"prediction[batch_idx].shape: {prediction[batch_idx].shape}")
            # print(f"target[batch_idx].shape: {target[batch_idx].shape}")
            pred = prediction[batch_idx].permute(1, 2, 0)  # [H, W, 3]
            tgt = target[batch_idx].permute(1, 2, 0)       # [H, W, 3]
            
            # Debug: print ranges of pred and tgt
            # print(f"pred range: [{pred.min().item():.2f}, {pred.max().item():.2f}]")
            # print(f"tgt range: [{tgt.min().item():.2f}, {tgt.max().item():.2f}]")
            
            # Create masks - fix forgery mask to match FORGERY_COLOR
            bg_mask = (tgt == AUTHENTIC_COLOR).all(dim=-1)  # Authentic regions
            fg_mask = (tgt == FORGERY_COLOR).all(dim=-1)    # Forged regions (white)
            
            loss = 0.0
            valid_losses = 0
            
            # 1. Authentic region loss - push toward black
            if bg_mask.any():
                bg_pred = pred[bg_mask].mean(dim=0)
                # print(f"bg_pred.shape: {bg_pred.shape}")
                loss_bg = F.huber_loss(
                    bg_pred, 
                    AUTHENTIC_COLOR, 
                    reduction='mean'
                )
                loss += loss_bg
                self._blackLoss += loss_bg.item()
                valid_losses += 1
                
            # 2. Forgery region loss - push toward white
            if fg_mask.any():
                fg_pred = pred[fg_mask].mean(dim=0)
                loss_fg = F.huber_loss(
                    fg_pred, 
                    FORGERY_COLOR,
                    reduction='mean'
                )
                loss += loss_fg
                self._whiteLoss += loss_fg.item()
                valid_losses += 1
                
            # 3. Class separation loss (critical for subtle forgeries)
            if bg_mask.any() and fg_mask.any():
                mean_authentic = pred[bg_mask].mean(dim=0)
                mean_forgery = pred[fg_mask].mean(dim=0)
                
                # Squared Euclidean distance between class means
                mean_distance = torch.sum((mean_authentic - mean_forgery) ** 2)
                
                # Maximize separation (minimize 1/distance)
                # Using 1/(1+distance) to avoid exploding gradients
                separation_loss = 300.0 / (1.0 + mean_distance)
                loss += separation_loss
                self._separationLoss += separation_loss.item()
                valid_losses += 1
            
            # Normalize by number of active loss terms
            if valid_losses > 0:
                total_loss += loss / valid_losses
        
        # Return average loss across batch
        return total_loss / batch_size



import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Lovász hinge helpers
# ------------------------------
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # gradient is piecewise constant
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss for flat tensors
    logits: [P] (float)
    labels: [P] (0 or 1)
    """
    if len(labels) == 0:
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * signs)

    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def flatten_binary_scores(preds, labels):
    """
    Flattens predictions and labels to 1D
    """
    preds = preds.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    return preds, labels

# ------------------------------
# Main Lovasz loss class
# ------------------------------
class BinaryLovaszLoss(nn.Module):
    def __init__(self):
        super(BinaryLovaszLoss, self).__init__()
        self.name = "BinaryLovaszLoss"

    def forward(self, prediction, target):
        """
        Args:
            prediction: [B, 3, H, W] in [0, 255] (model output)
            target:     [B, 3, H, W] in [0, 255]
                        authentic=black (0), forgery=white (255)
        Returns:
            torch.Tensor: Scalar loss
        """
        device = prediction.device
        prediction = prediction.float() / 255.0  # scale to [0,1]
        target = target.float() / 255.0          # scale to [0,1]

        batch_size = prediction.size(0)
        losses = []

        for b in range(batch_size):
            pred_b = prediction[b]  # [3, H, W]
            tgt_b = target[b]       # [3, H, W]

            # Convert to binary mask using the first channel
            # Works because authentic=0, forgery=1 after scaling
            pred_gray = pred_b.mean(dim=0)  # [H, W]
            tgt_gray = tgt_b.mean(dim=0)    # [H, W]

            # Convert logits: center around 0 for hinge loss
            logits = (pred_gray * 2.0 - 1.0)  # range [-1, 1]
            labels = (tgt_gray > 0.5).float()

            logits_flat, labels_flat = flatten_binary_scores(logits, labels)
            loss_b = lovasz_hinge_flat(logits_flat, labels_flat)
            losses.append(loss_b)

        return torch.mean(torch.stack(losses))
