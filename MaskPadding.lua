require 'nn'
local MaskPadding, parent = torch.class('nn.MaskPadding', 'nn.Criterion')
function MaskPadding:__init(criterion, padVal)
  parent.__init(self)
  self.padVal = padVal or 0
  self.index = torch.Tensor()
  self.criterion = criterion
  self:type(criterion.gradInput:type())
end

function MaskPadding:reduceReshape(input, target)
  local target = target:contiguous():view(-1,1)
  self.index = self.index:long()
  self.index:resize(target:size(1)):range(1, target:size(1))
  self.index = self.index:maskedSelect(target:ne(self.padVal):byte())
  self.reducedTarget = target:view(-1):index(1,self.index)
  self.reducedInput = input:view(-1, input:size(input:dim())):index(1,self.index)
end

function MaskPadding:updateOutput(input, target)
  self:reduceReshape(input,target)
  return self.criterion:updateOutput(self.reducedInput, self.reducedTarget)
end
function MaskPadding:updateGradInput(input, target)
  if not self.reducedInput then self:reduceReshape(input, target) end
  local gradOutput = self.criterion:updateGradInput(self.reducedInput, self.reducedTarget)
  self.gradInput:resizeAs(input):zero()
  self.gradInput:view(-1, input:size(input:dim())):indexCopy(1, self.index, gradOutput:view(-1,gradOutput:size(gradOutput:dim())))
  return self.gradInput
end
----
--function MaskPadding:parameters()
--  return self.noiser:parameters()
--end
