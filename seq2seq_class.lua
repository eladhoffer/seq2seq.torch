require 'nn'
require 'recurrent'

local seq2seq = torch.class('seq2seq')
function seq2seq:__init(...)
  local args = dok.unpack(
  {...},
  'seq2seq',
  'Initializes a seq2seq ',
  {arg='encoder', type='table', help='encoder model'},
  {arg='maxLength', type='number', help='maximum sequence length', default = 50}
  )
  self.encoderModel = args.encoder
  self.maxLength = args.maxLength
  self.decoderModels = {}
  self.allModules = nn.Sequential():add(self.encoderModel)
end

function seq2seq:getParameters()
  return self.allModules:getParameters()
end

function seq2seq:type(type)
  self.allModules:type(type)
  return self
end

function seq2seq:cuda()
  return seq2seq:type('torch.CudaTensor')
end

function seq2seq:float()
  return seq2seq:type('torch.FloatTensor')
end

function seq2seq:addDecoder(model)
  local num = #self.decoderModels + 1
  self.decoderModels[num] = model
  self.allModules:add(model)
  return self
end

function seq2seq:training()
  self.train = true
  self.allModules:training()
end

function seq2seq:forward(input)
  --input is of form {input, decOutput1, decOutput2, ...}
  self.allModules:sequence()
  self.allModules:forget()
  self.allModules:setIterations(self.maxLength)

  self.encoderModel:zeroState()
  local y = {}
  y[1] = self.encoderModel:forward(input[1])
  local state = self.encoderModel:getState()

  for m = 2, #input do
    self.decoderModels[m - 1]:setState(state)
    y[m] = self.decoderModels[m - 1]:forward(input[m])
  end

  return y
end

function seq2seq:backward(input, gradOutput)
  --input is of form {input, decOutput1, decOutput2, ...}
  self.encoderModel:zeroGradState()
  local gradInput = {}
  for m = 2, #input do
    self.decoderModels[m - 1]:zeroGradState()
    gradInput[m] = self.decoderModels[m - 1]:backward(input[m], gradOutput[m])
    local decoderGradState = self.decoderModels[m - 1]:getGradState()
    self.encoderModel:accGradState(decoderGradState)
  end
  gradInput[1] = self.encoderModel:backward(input[1], gradOutput[1])
  return gradInput
end

function seq2seq:training()
  self.allModules:training()
end

function seq2seq:evaluate()
  self.allModules:evaluate()
end
