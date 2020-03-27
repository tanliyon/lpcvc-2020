import torch
import torchvision
from torch import nn, optim

lambda_score = 1
lambda_geometry = 1

class Loss(nn.Module):
    def __init__(self, weight_angle=10):
        super(Loss, self).__init__()
        self.weight_angle = weight_angle

    # def compute_geometry_beta(self, y_truth_vertices):
    #
    #     D = []
    #     for i in range(0, 8, 2): # 0,2,4,6
    #         indices = [i, i+1, (i+2)%8, (i+3)%8]
    #         x1, y1, x2, y2 = y_true_geometry_cell[indices]
    #         d = (x1 - x2) ** 2 + (y1 - y2) ** 2
    #         D.append(d)
    #     D = torch.Tensor(D)
    #
    #     return torch.sqrt(torch.min(D))
    #
    #
    # def compute_score_loss(self, Y_true_score, Y_pred_score):
    #
    #     """
    #     y_true_score, y_pred_score: [m, 1, 128, 128]; range: [0,1]
    #     """
    #
    #     m = Y_true_score.shape[0]
    #     n_cells = torch.numel(Y_true_score)
    #     n_pos_cells = Y_true_score.sum()
    #     n_neg_cells = n_cells - n_pos_cells
    #     beta = 1 - (Y_true_score.sum()/torch.numel(Y_true_score)) # ratio of 0s
    #     loss_of_score_pos = -beta * Y_true_score * torch.log(Y_pred_score) # [m, 1, 128, 128]
    #     loss_of_score_neg = -(1 - beta) * (1 - Y_true_score) * torch.log(1 - Y_pred_score) # [m, 1, 128, 128]
    #     normalization_factor = (beta * n_pos_cells) + ((1-beta)* n_neg_cells)
    #     loss_of_score = torch.sum(loss_of_score_pos + loss_of_score_neg) / normalization_factor
    #
    #     return loss_of_score
    #
    #
    # def compute_geometry_loss(self, Y_true_geometry, Y_pred_geometry, Y_true_score, smoothed_l1_loss_beta=1):
    #
    #     """
    #     Y_true_geometry, Y_pred_geometry: [m, 8, 128, 128]; range:[0,1]
    #     beta: N_Q*
    #     Y_true_score: [m, 1, 128, 128]
    #     """
    #     beta = smoothed_l1_loss_beta
    #     diff = torch.abs(Y_true_geometry*Y_true_score - Y_pred_geometry*Y_true_score) # multiply with text mask
    #     diff = diff / 512
    #     #diff = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    #     loss_of_geometry = diff.sum()
    #     loss_of_geometry /= float(Y_true_score.sum()*8)
    #
    #     return loss_of_geometry
    #
    # def compute_loss(self, Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry, smoothed_l1_loss_beta=1):
    #     """
    #     y_true_score, y_pred_score: [m, 1, 128, 128]
    #     y_true_geometry, y_pred_geometry: [m, 8, 128, 128]
    #     """
    #     #print("Y_true_geometry.max():", torch.max(Y_true_geometry).item())
    #     #print("Y_pred_geometry.max():", torch.max(Y_pred_geometry).item())
    #     self.loss_of_score = self.compute_score_loss(Y_true_score, Y_pred_score)
    #     self.loss_of_geometry = self.compute_geometry_loss(Y_true_geometry,
    #                                                        Y_pred_geometry,
    #                                                        Y_true_score,
    #                                                        smoothed_l1_loss_beta=smoothed_l1_loss_beta)
    #     self.loss = lambda_score * self.loss_of_score + lambda_geometry * self.loss_of_geometry
    #     return self.loss

    def get_dice_loss(self, gt_score, pred_score):
    	inter = torch.sum(gt_score * pred_score)
    	union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    	return 1. - (2 * inter / union)


    def get_geo_loss(self, gt_geo, pred_geo):
    	d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    	d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
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
        # if torch.sum(geometry_map) < 1:
        #     return torch.sum(predicted_score_map + predicted_geometry_map) * 0
        #
        # loss = self.compute_loss(score_map, predicted_score_map, geometry_map, predicted_geometry_map)
        # return loss
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0

        classify_loss = self.get_dice_loss(gt_score, pred_score*(1-ignored_map))
        iou_loss_map, angle_loss_map = self.get_geo_loss(gt_geo, pred_geo)

        angle_loss = torch.sum(angle_loss_map*gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map*gt_score) / torch.sum(gt_score)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss, iou_loss))
        return geo_loss + classify_loss

# test code
if __name__ == '__main__':
    loss_function = Loss()
    Y_true_score = torch.rand([1, 128, 128])
    Y_pred_score = torch.rand([1, 1, 128, 128])
    Y_true_geometry = torch.rand([8, 128, 128])
    Y_pred_geometry = torch.rand([1,8, 128, 128])
    loss = loss_function(Y_true_score, Y_pred_score, Y_true_geometry, Y_pred_geometry)
    print("Score Loss:", loss_function.loss_of_score)
    print("Geometry Loss:", loss_function.loss_of_geometry)
    print("Loss:", loss)
