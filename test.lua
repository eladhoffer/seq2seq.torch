require 'MaskPadding'
require 'cunn'
m = nn.MaskPadding(nn.CrossEntropyCriterion()):cuda()

y=torch.rand(3,4):cuda():fill(1)
y[{1,{3,4}}]:fill(0)
y[3][4]=0

x = torch.rand(3,4,5):cuda()
z=m:forward(x,y)

m:backward(x,z)
