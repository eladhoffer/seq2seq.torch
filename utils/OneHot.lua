require 'nn'
local OneHot, parent = torch.class('nn.OneHot', 'nn.Module')

function OneHot:__init(outputSize, zeroOnNum)
  parent.__init(self)
  self.outputSize = outputSize
  self.zeroOnNum = zeroOnNum
end

function OneHot:updateOutput(input)
  local input_ = input:contiguous()
  if input:dim() == 1 or input:size(2)>1 then
    input_ = input_:view(-1,1)
  end
  self.output:resize(input_:size(1), self.outputSize):zero()
  if torch.type(input_) == 'torch.CudaTensor' then
    self.output:scatter(2, input_, 1)
  else
    self.output:scatter(2, input_:long(), 1)
  end
  if input:dim() == 1 then
    self.output = self.output:view(input:size(1), -1)
  elseif input:size(2)>1 then
    self.output = self.output:view(input:size(1), input:size(2), -1)
  end
  if self.zeroOnNum then
    self.output:select(self.output:dim(), self.zeroOnNum):zero()
  end
  return self.output
end

function OneHot:clone(...)
  local m = parent.clone(self, ...)
  m.zeroOnNum = self.zeroOnNum
  return m
end

function OneHot:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  return self.gradInput
end
