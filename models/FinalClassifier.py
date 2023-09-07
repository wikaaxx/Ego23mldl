from torch import nn
from torch.nn.init import *
from torch.autograd import Function
from models.TRNmodule import RelationModuleMultiScale
import torch.nn.functional as F

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None
class Classifier(nn.Module):
    def __init__(self, num_class, num_segments=5, feature_dim=1024, share_params='Y', add_fc=1,
                 frame_aggregation='none', use_attn=False, beta=[1,1,1], dropout_i=0.5, dropout_v=0.5,
                 ens_DA=None):
        super().__init__()
        self.num_segments = num_segments
        self.feature_dim = feature_dim
        self.share_params = share_params
        self.add_fc = add_fc
        self.frame_aggregation = frame_aggregation
        self.use_attn = use_attn
        self.beta = beta
        self.dropout_rate_i = dropout_i
        self.dropout_rate_v = dropout_v
        self.ens_DA = ens_DA
        self.before_softmax = True
        self._prepare_DA(num_class)
    def _prepare_DA(self, num_class):  # convert the model to DA framework
        std = 0.001
        self.softmax = nn.Softmax
        self.relu = nn.ReLU(inplace=True)
        self.dropout_i = nn.Dropout(p=self.dropout_rate_i)
        self.dropout_v = nn.Dropout(p=self.dropout_rate_v)
        # ------ FRAME-LEVEL layers (shared layers + source layers + domain layers) ------#
        # 1. shared feature layers
        self.fc_feature_shared_source = nn.Linear(self.feature_dim, self.feature_dim)
        normal_(self.fc_feature_shared_source.weight, 0, std)
        constant_(self.fc_feature_shared_source.bias, 0)
        if self.add_fc > 1:
            self.fc_feature_shared_2_source = nn.Linear(self.feature_dim, self.feature_dim)
            normal_(self.fc_feature_shared_2_source.weight, 0, std)
            constant_(self.fc_feature_shared_2_source.bias, 0)
        if self.add_fc > 2:
            self.fc_feature_shared_3_source = nn.Linear(self.feature_dim, self.feature_dim)
            normal_(self.fc_feature_shared_3_source.weight, 0, std)
            constant_(self.fc_feature_shared_3_source.bias, 0)
        # 3. domain feature layers (frame-level)
        self.fc_feature_domain = nn.Linear(self.feature_dim, self.feature_dim)
        normal_(self.fc_feature_domain.weight, 0, std)
        constant_(self.fc_feature_domain.bias, 0)
        self.fc_classifier_domain = nn.Linear(self.feature_dim, 2)
        normal_(self.fc_classifier_domain.weight, 0, std)
        constant_(self.fc_classifier_domain.bias, 0)
        if self.share_params == 'N':
            self.fc_feature_shared_target = nn.Linear(self.feature_dim, self.feature_dim)
            normal_(self.fc_feature_shared_target.weight, 0, std)
            constant_(self.fc_feature_shared_target.bias, 0)
            if self.add_fc > 1:
                self.fc_feature_shared_2_target = nn.Linear(self.feature_dim, self.feature_dim)
                normal_(self.fc_feature_shared_2_target.weight, 0, std)
                constant_(self.fc_feature_shared_2_target.bias, 0)
            if self.add_fc > 2:
                self.fc_feature_shared_3_target = nn.Linear(self.feature_dim, self.feature_dim)
                normal_(self.fc_feature_shared_3_target.weight, 0, std)
                constant_(self.fc_feature_shared_3_target.bias, 0)
        # ------ AGGREGATE FRAME-BASED features (frame feature --> video feature) ------#
        if self.frame_aggregation == 'trn':  # TRN multiscale
            self.num_bottleneck = 256  # or 512
            self.TRN = RelationModuleMultiScale(self.feature_dim, self.num_bottleneck, self.num_segments)
            self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
            self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)
        # ------ VIDEO-LEVEL layers (source layers + domain layers) ------#
        # if self.frame_aggregation == 'avgpool'
        feat_aggregated_dim = self.feature_dim
        if self.frame_aggregation == 'trn':
            feat_aggregated_dim = self.num_bottleneck
        # 2. domain feature layers (video-level)
        self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_aggregated_dim)
        normal_(self.fc_feature_domain_video.weight, 0, std)
        constant_(self.fc_feature_domain_video.bias, 0)
        # 3. classifiers (video-level)
        self.fc_classifier_video_source = nn.Linear(feat_aggregated_dim, num_class)
        normal_(self.fc_classifier_video_source.weight, 0, std)
        constant_(self.fc_classifier_video_source.bias, 0)
        
        self.fc_classifier_domain_video = nn.Linear(feat_aggregated_dim, 2)
        normal_(self.fc_classifier_domain_video.weight, 0, std)
        constant_(self.fc_classifier_domain_video.bias, 0)
        # domain classifier for TRN
        if self.frame_aggregation == 'trn':  
            self.relation_domain_classifier_all = nn.ModuleList()
            for i in range(self.num_segments - 1):
                relation_domain_classifier = nn.Sequential(
                    nn.Linear(feat_aggregated_dim, feat_aggregated_dim),
                    nn.ReLU(),
                    nn.Linear(feat_aggregated_dim, 2)
                )
                self.relation_domain_classifier_all += [relation_domain_classifier]
        if self.share_params == 'N':
            self.fc_classifier_video_target = nn.Linear(feat_aggregated_dim, num_class)
            normal_(self.fc_classifier_video_target.weight, 0, std)
            constant_(self.fc_classifier_video_target.bias, 0)

    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy

        return weights

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        weights_attn = self.get_trans_attn(pred_domain)
        weights_attn = weights_attn.view(-1, num_segments - 1, 1).repeat(1, 1, feat_fc.size()[-1])  # reshape & repeat weights (e.g. 16 x 4 x 256)
        feat_fc_attn = (weights_attn + 1) * feat_fc

        return feat_fc_attn, weights_attn[:, :, 0]

    # Gsd
    def domain_classifier_frame(self, feat, beta):
        feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
        feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
        feat_fc_domain_frame = self.relu(feat_fc_domain_frame)

  
        pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)
        return pred_fc_domain_frame
    # Gtd
    def domain_classifier_video(self, feat_video, beta):
        feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
        feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
        feat_fc_domain_video = self.relu(feat_fc_domain_video)
        pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)
        return pred_fc_domain_video
    # Grd
    def domain_classifier_relation(self, feat_relation, beta):
        # 128x4x256 --> (128x4)x2
        pred_fc_domain_relation_video = None
        for i in range(len(self.relation_domain_classifier_all)):
            feat_relation_single = feat_relation[:, i, :].squeeze(1)  # 128x1x256 --> 128x256
            feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single,
                                                               beta[0])  # the same beta for all relations (for now)
            pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
            if pred_fc_domain_relation_video is None:
                pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1, 1, 2)
            else:
                pred_fc_domain_relation_video = torch.cat(
                    (pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1, 1, 2)), 1)
        pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1, 2)
        return pred_fc_domain_relation_video
    # AvgPoool
    def aggregate_frames(self, feat_fc, num_segments, pred_domain):
        feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
        if self.use_attn:  # get the attention weighting
            weights_attn = self.get_trans_attn(pred_domain)
            weights_attn = weights_attn.view(-1, 1, num_segments, 1).repeat(1, 1, 1, feat_fc.size()[-1])  # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
            feat_fc_video = (weights_attn + 1) * feat_fc_video

        feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
  
        feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512
        return feat_fc_video
    def forward(self, input_source, input_target):
        # batch_source, batch_target
        batch_source = input_source.size()[0]
        batch_target = input_target.size()[0]
        # initailize dicts
        pred_domain_all_source = {}
        pred_domain_all_target = {}
        # input_data is a list of tensors --> need to do pre-processing
        feat_base_source = input_source.view(-1, input_source.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
        # === MLP === with 1 lvl
        feat_fc_source = self.fc_feature_shared_source(feat_base_source)
        feat_fc_target = self.fc_feature_shared_target(feat_base_target) if self.share_params == 'N' else self.fc_feature_shared_source(feat_base_target)
        feat_fc_source = self.relu(feat_fc_source)
        feat_fc_target = self.relu(feat_fc_target)
        feat_fc_source = self.dropout_i(feat_fc_source)
        feat_fc_target = self.dropout_i(feat_fc_target)
        if self.add_fc > 1:
            feat_fc_source = self.fc_feature_shared_2_source(feat_fc_source)
            feat_fc_target = self.fc_feature_shared_2_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_2_source(feat_fc_target)
            feat_fc_source = self.relu(feat_fc_source)
            feat_fc_target = self.relu(feat_fc_target)
            feat_fc_source = self.dropout_i(feat_fc_source)
            feat_fc_target = self.dropout_i(feat_fc_target)
        if self.add_fc > 2:
            feat_fc_source = self.fc_feature_shared_3_source(feat_fc_source)
            feat_fc_target = self.fc_feature_shared_3_target(feat_fc_target) if self.share_params == 'N' else self.fc_feature_shared_3_source(feat_fc_target)
            feat_fc_source = self.relu(feat_fc_source)
            feat_fc_target = self.relu(feat_fc_target)
            feat_fc_source = self.dropout_i(feat_fc_source)
            feat_fc_target = self.dropout_i(feat_fc_target)
        # === adversarial branch (frame-level) (GSD) === #
        pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, self.beta)
        pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, self.beta)
        pred_domain_all_source['GSD'] = pred_fc_domain_frame_source.view((batch_source, self.num_segments) + pred_fc_domain_frame_source.size()[-1:])
        pred_domain_all_target['GSD'] = pred_fc_domain_frame_target.view((batch_target, self.num_segments) + pred_fc_domain_frame_target.size()[-1:])
        ### aggregate the frame-based features to video-based features ###
        if self.frame_aggregation == 'avgpool':
            feat_fc_video_source = self.aggregate_frames(feat_fc_source, self.num_segments, pred_fc_domain_frame_source)
            feat_fc_video_target = self.aggregate_frames(feat_fc_target, self.num_segments, pred_fc_domain_frame_target)
        elif self.frame_aggregation == 'trn':
            feat_fc_video_source = feat_fc_source.view((-1, self.num_segments) + feat_fc_source.size()[-1:])
            # reshape based on the segments (e.g. 640x512 --> 128x5x512)
            feat_fc_video_target = feat_fc_target.view((-1, self.num_segments) + feat_fc_target.size()[-1:])
            # reshape based on the segments (e.g. 640x512 --> 128x5x512)
            feat_fc_video_relation_source = self.TRN(feat_fc_video_source)  # 128x5x512 --> 128x4x256 (256-dim. relation feature vectors x 5)
            feat_fc_video_relation_target = self.TRN(feat_fc_video_target)
            # adversarial branch GRD
            pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, self.beta)
            pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, self.beta)
            num_relation = feat_fc_video_relation_source.size()[1]
            pred_domain_all_source['GRD'] = pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:])
            pred_domain_all_target['GRD'] = pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:])
            # transferable attention
            if self.use_attn:  # get the attention weighting
                feat_fc_video_relation_source, _ = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, self.num_segments)
                feat_fc_video_relation_target, _ = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, self.num_segments)

            # sum up relation features (ignore 1-relation)
            feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)

  
            feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)
        else:
            raise NotImplementedError
        # === source layers (video-level) ===#
        feat_fc_video_source = self.dropout_v(feat_fc_video_source)
        feat_fc_video_target = self.dropout_v(feat_fc_video_target)
        pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
        pred_fc_video_target = self.fc_classifier_video_target(feat_fc_video_target) if self.share_params == 'N' else self.fc_classifier_video_source(feat_fc_video_target)
        # === adversarial branch (video-level) (GVD) === #
        pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, self.beta)
        pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, self.beta)

        pred_domain_all_source['GVD'] = pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:])
        pred_domain_all_target['GVD'] = pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:])

        # === final output ===#
        if not self.before_softmax:
  
            pred_fc_video_source = self.softmax(pred_fc_video_source)
            pred_fc_video_target = self.softmax(pred_fc_video_target)
        
        #feat_afn=feat_fc_video_source.clone()
        return pred_fc_video_source, pred_domain_all_source, pred_fc_video_target, pred_domain_all_target #,normal_(feat_afn, 0, 0.001)
        # Ly: pred_fc_video_source
        # Lsd, Lrd, Lvd: pred_domain_all_source, pred_domain_all_target
        # Lae: pred_fc_video_source, pred_domain_all_source, pred_fc_video_target, pred_domain_all_target