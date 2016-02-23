local ISCriterion.lua, Criterion = torch.class('nn.ISCriterion.lua', 'nn.Criterion')

function ISCriterion.lua:__init(weights)
   Criterion.__init(self)
   self.lsm = nn.LogSoftMax()
   self.nll = nn.ClassNLLCriterion(weights)
   self.join = nn.JoinTable(1,2)
self.concat = nn.Concat
   nn.View(-1, 1):setNumInputDims(1)
   nn.View(1, -1)
end

function ISCriterion.lua:updateOutput(input, target, noiseSamples)
   input = input:squeeze()
   target = nn.utils.addSingletonDimension(target, 2)
   noiseSamples = nn.utils.addSingletonDimension(noiseSamples, 1):expand(target:size(1), 1, 1)
   self.join:updateOutput({target, noiseSamples})
   self.lsm:updateOutput(input)
   self.nll:updateOutput(self.lsm.output, target)
   self.output = self.nll.output
   return self.output
end

function ISCriterion.lua:updateGradInput(input, target)
   local size = input:size()
   input = input:squeeze()
   target = type(target) == 'number' and target or target:squeeze()
   self.nll:updateGradInput(self.lsm.output, target)
   self.lsm:updateGradInput(input, self.nll.gradInput)
   self.gradInput:view(self.lsm.gradInput, size)
   return self.gradInput
end

return nn.ISCriterion.lua
