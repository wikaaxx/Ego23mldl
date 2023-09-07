from abc import ABC
import torch
from torch import nn
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger
from typing import Dict, Tuple
import torch.nn.functional as F
class ActionRecognition(tasks.Task, ABC):
    """Action recognition model."""
    
    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int, 
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args, **kwargs) -> None:
        """Create an instance of the action recognition model.
        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args
        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        self.loss = utils.AverageMeter()
        
        self.num_clips = num_clips
        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        
        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)
    def forward(self, input_source: Dict[str, torch.Tensor], input_target: Dict[str, torch.Tensor] = None, **kwargs):
        """Forward step of the task
        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality
        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """
        if input_target is None:
            input_target = input_source
        pred_fc_video_source = {}
        pred_fc_video_target = {}
        pred_domain_source = {}
        pred_domain_target = {}
        for i_m, m in enumerate(self.modalities):
          #feat_atn dodaj
            pred_fc_video_source[m], pred_domain_all_source, pred_fc_video_target[m], pred_domain_all_target = self.task_models[m](input_source[m], input_target[m], **kwargs)
            if i_m == 0:
                for k in pred_domain_all_source.keys():
                    pred_domain_source[k] = {}
                for k in pred_domain_all_target.keys():
                    pred_domain_target[k] = {}
            for k in pred_domain_all_source.keys():
                pred_domain_source[k][m] = pred_domain_all_source[k]
            for k in pred_domain_all_target.keys():
                pred_domain_target[k][m] = pred_domain_all_target[k]
                #dodaj do return feat_afn
        return pred_fc_video_source, pred_domain_source, pred_fc_video_target, pred_domain_target #, feat_afn
    def get_entropy_attn(self, feat):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(feat) * logsoftmax(feat), 1)
        return entropy
    
    #dodaj feat_afn
    def compute_loss(self, logits_source, source_label, pred_domain_source, logits_target, pred_domain_target):
        
        
        modality = 'RGB'
        adversarial_loss = {}

        loss = 0  # loss totale

        # Loss su Grd, Gvd, Gsd
        for k in pred_domain_source.keys():
            pred_domain_source_single = pred_domain_source[k][modality].view(-1, pred_domain_source[k][modality].size(-1)) #was without .size just [-1]
            pred_domain_target_single = pred_domain_target[k][modality].view(-1, pred_domain_target[k][modality].size(-1))
            # 32x5x2 -> 160x2

            source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
            target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()

            domain_label = torch.cat((source_domain_label, target_domain_label), 0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            domain_label = domain_label.to(device)

            pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single), 0)

            adversarial_loss_single = self.criterion(pred_domain, domain_label)
            
            adversarial_loss[k] = torch.mean(adversarial_loss_single) * self.model_args[modality].weight[k]
            loss += adversarial_loss[k]

        # Loss y (classi)
        fused_logits_source = reduce(lambda x, y: x + y, logits_source.values())  
        
        classification_loss = torch.mean(self.criterion(fused_logits_source, source_label))
        loss += classification_loss
        #hafn loss
        #radius = 1 #0.1 0.5,1,5,25, HAFN RADIUS
        #weights = 0.01
        #cls_loss= F.nll_loss(F.log_softmax(fused_logits_source), source_label)
        #l = (feat_afn.norm(p=2, dim=1).mean() - radius) ** 2
        #l1=weights*l
        #hafn_loss = l1+cls_loss
        #loss+= hafn_loss

        #safn
        #radius1 = 0.05  #0.1 0.3 0.5, SAFN radius
        #cls_loss= F.nll_loss(F.log_softmax(fused_logits_source), source_label)
        #radius2 = feat_afn.norm(p=2, dim=1).detach()
        #radius2 = radius2 + radius1
        #l2 = ((feat_afn.norm(p=2, dim=1) - radius2) ** 2).mean()
        #l3=weights*l2
        #safn_loss=l3 + cls_loss
        #loss+=safn_loss

        # Loss attn
        fused_logits_target = reduce(lambda x, y: x + y, logits_target.values())  # somma sulle modalità
        fused_logits = torch.cat((fused_logits_source, fused_logits_target), 0)
        # 64x8

        pred_domain_source_gvd = pred_domain_source['GVD'][modality].view(-1, pred_domain_source[k][modality].size(-1))
        pred_domain_target_gvd = pred_domain_target['GVD'][modality].view(-1, pred_domain_target[k][modality].size(-1)) #again was [-1]

        pred_domain_attn = torch.cat((pred_domain_source_gvd, pred_domain_target_gvd), 0)

        Hy = self.get_entropy_attn(fused_logits)
        Hd = self.get_entropy_attn(pred_domain_attn)
  
        attn_loss_single = (1 + Hd) * Hy
        attn_loss = torch.mean(attn_loss_single) * self.model_args[modality].weight['Attn']
        loss += attn_loss
        
        # Update the loss value, weighting it by the ratio of the batch size to the total
        # batch size (for gradient accumulation)
        self.loss.update(loss / (self.total_batch / self.batch_size), self.batch_size)
        return loss

    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.
        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        fused_logits = reduce(lambda x, y: x + y, logits.values())
        self.accuracy.update(fused_logits, label)
    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.loss.val, 
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }
        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)
    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr
            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")
    def reset_loss(self):
        """Reset the classification loss.
        This method must be called after each optimization step.
        """
        self.loss.reset()
    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()
    def step(self):
        """Perform an optimization step.
        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()
    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.
        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.
        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        self.loss.val.backward(retain_graph=retain_graph)