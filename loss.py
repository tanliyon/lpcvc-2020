import torch
import torchvision
from torch import nn, optim

class Loss(nn.Module):
    # weight_angle = 10
    def __init__(self, weight_angle=20):
        super(Loss, self).__init__()
        self.weight_angle = weight_angle

    # def get_dice_loss(self, gt_score, pred_score):
    def get_dice_loss(self, gt_score, pred_score, ignored_map):
    	# inter = torch.sum(gt_score * pred_score)
    	# union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
        inter = torch.sum(gt_score * pred_score * ignored_map)
        union = torch.sum(gt_score * ignored_map) + torch.sum(pred_score * ignored_map) + 1e-5
        return 1. - (2 * inter / union)

    def get_geo_loss(self, gt_geo, pred_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 0)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 0)

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)

        w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        iou_loss_map = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
        angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
        return iou_loss_map, angle_loss_map

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        # classify_loss = self.get_dice_loss(gt_score, pred_score*(1-ignored_map))
        classify_loss = self.get_dice_loss(gt_score, pred_score, 1 - ignored_map)
        iou_loss_map, angle_loss_map = self.get_geo_loss(gt_geo, pred_geo)

        # angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        # iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)
        # geo_loss = self.weight_angle * angle_loss + iou_loss
        geo_loss = self.weight_angle * angle_loss_map + iou_loss_map
        # return geo_loss + classify_loss
        return torch.mean(geo_loss * (1 - ignored_map) * gt_score) + classify_loss

# test code
if __name__ == '__main__':
    loss_function = Loss()
    Y_true_score = torch.rand([1, 128, 128])
    Y_pred_score = torch.rand([1, 1, 128, 128])
    Y_true_geometry = torch.rand([8, 128, 128])
    Y_pred_geometry = torch.rand([1, 8, 128, 128])
    loss = loss_function(Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry)
    print("Score Loss:", loss_function.loss_of_score)
    print("Geometry Loss:", loss_function.loss_of_geometry)
    print("Loss:", loss)
