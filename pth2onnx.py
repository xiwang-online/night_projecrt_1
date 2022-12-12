
import model
import torchvision
import onnx
import torch
from torch.autograd import Variable


input_name = ['input']
output_name = ['output']
input = Variable(torch.randn(1, 3, 512, 512))#.cuda()
net = model.PhysicalNN()#.cuda()
checkpoint = torch.load('./checkpoints/model_best_354.pth',map_location='cpu')
net.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
#print(checkpoint['state_dict'])
#net.load_state_dict(checkpoint['state_dict'])
torch.onnx.export(net, input, 'night_1130test.onnx', input_names=input_name, output_names=output_name, verbose=True, opset_version=12)
# 测试模型是否转换成功
test = onnx.load('night_1130test.onnx')
onnx.checker.check_model(test)
print("==> Passed")


