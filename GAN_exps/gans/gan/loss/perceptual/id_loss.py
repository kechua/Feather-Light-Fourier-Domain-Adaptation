import torch
from torch import nn

from nn.resnet.ir_se import make_res_blocks, bottleneck_IR, bottleneck_IR_SE


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
	norm = torch.norm(input, 2, axis, True)
	output = torch.div(input, norm)
	return output


class Backbone(nn.Module):
	def __init__(self, input_size, num_layers, mode='ir', drop_ratio=0.4, affine=True):
		super(Backbone, self).__init__()
		assert input_size in [112, 224], "input_size should be 112 or 224"
		assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
		assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
		blocks = make_res_blocks(num_layers)
		if mode == 'ir':
			unit_module = bottleneck_IR
		elif mode == 'ir_se':
			unit_module = bottleneck_IR_SE
		self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
									  nn.BatchNorm2d(64),
									  nn.PReLU(64))
		if input_size == 112:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               Flatten(),
			                               nn.Linear(512 * 7 * 7, 512),
			                               nn.BatchNorm1d(512, affine=affine))
		else:
			self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
			                               nn.Dropout(drop_ratio),
			                               Flatten(),
			                               nn.Linear(512 * 14 * 14, 512),
			                               nn.BatchNorm1d(512, affine=affine))

		modules = []
		for block in blocks:
			for bottleneck in block:
				modules.append(unit_module(bottleneck.in_channel,
										   bottleneck.depth,
										   bottleneck.stride))
		self.body = nn.Sequential(*modules)

	def forward(self, x):
		x = self.input_layer(x)
		x = self.body(x)
		x = self.output_layer(x)
		return l2_norm(x)


class IDLoss(nn.Module):
    def __init__(self, model_path_ir_se50):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_path_ir_se50))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
