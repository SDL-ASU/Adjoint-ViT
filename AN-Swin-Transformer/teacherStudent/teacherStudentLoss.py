import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherStudentLoss(nn.Module):
	def __init__(self, training=False):
		super().__init__()
		self.training = training

	def forward(self, teacherOutput, studentOutput, target):
		if self.training:
			loss = torch.sum(-target * F.log_softmax(studentOutput, dim=-1), dim=-1)
			loss = loss.mean()

			prob1 = F.softmax(teacherOutput, dim=-1)
			prob2 = F.softmax(studentOutput, dim=-1)
			kl = (prob1 * torch.log(1e-6 + prob1/(prob2+1e-6))).sum(1)

			return loss + kl.mean()
		else:
			log_preds = F.log_softmax(studentOutput, dim=-1)
			loss = F.nll_loss(log_preds, target)

			return loss
