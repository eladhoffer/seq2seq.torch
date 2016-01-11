require 'nn'
require 'optim'
local seq2seq = torch.class('seq2seq')
function seq2seq:__init(...)
  local args = dok.unpack(
  {...},
  'Initializes a seq2seq ',
  {arg='encoder', type='table', help='encoder model'},
  {arg='encoderVocab', type='table', help='encoder vocabulary table'},
  {arg='inputData', type='userdata', help='data to encode - tds vector or table'},
  {arg='type', type='string', help='Type of Tensor', default = 'torch.CudaTensor'},
  {arg='maxLength', type='number', help='maximum sequence length', default = 50},
  {arg='batchSize', type='number', help='batch size for training', default = 50},
  {arg='criterion', type='table', help='criterion used for training', default = nn.CrossEntropyCriterion()},
  {arg='optimization', type='table', help='optimization func used for training', default = optim.adam},
  {arg='optimConfig', type='table', help='optimization meta-parameters',
    default = {learningRate = 1e-3, gradClip = 5}}
  )

  for x,val in pairs(args) do
    self[x] = val
  end
  self.outputData = {}
  self.decoderModels = {}
  self.decoderVocabs = {}
  self.optimConfig.gradClip = self.optimConfig.gradClip or 5
end

function seq2seq:flattenParameters()
  self.allModules = nn.Sequential():add(self.encoderModel)
  for _, m in pairs(self.decoderModels) do
    self.allModules:add(m)
  end
  self.weights, self.gradients = self.allModules:getParameters()
end

function seq2seq:type(type)
  self.type = type
  self.encoderModel:type(self.type)
  for _, m in pairs(self.decoderModels) do
    m:type(self.type)
  end
  self.criterion = self.criterion:type(type)
  self:flattenParameters()
  return self
end

function seq2seq:cuda()
  return seq2seq:type('torch.CudaTensor')
end

function seq2seq:float()
  return seq2seq:type('torch.FloatTensor')
end

function seq2seq:addDecoder(model, data, vocab)
  assert(#self.inputData == #data, 'num of decoded sequences must match num encoded')
  local num = #self.decoderModels + 1
  self.outputData[num] = data
  self.decoderModels[num] = model
  self.decoderVocabs[num] = vocab
  return self
end

function seq2seq:training()
  self.train = true
  self.allModules:training()
end

function seq2seq:forward(input)
  --input is a table of {input, decOutput1, decOutput2, ...}
  local xE = table.remove(input, 1)
  local xDs = input
  self.allModules:sequence()
  self.allModules:forget()
  self.allModules:setIterations(self.maxLength)

  self.encoderModel:zeroState()
  local out = self.encoderModel:forward(xE)
  local state = self.encoderModel:getState()

  for m = 1, #decoderModels do
    self.decoderModels[m]:setState(state)
    y[m] = self.decoderModels[m]:forward(x[m])
    currLoss = lossNoPadding(criterion, y[m], yt[m], self.decoderVocabs[m]['<PAD>'])
    --  print(torch.exp(currLoss))
    lossVals[m] = lossVals[m] + currLoss --/ opt.seqLength
    seqCriterion:forward(y[m],yt[m])

  end

----------------------------------------------------------------------
function seq2seq:train()
  -- input is of form {data, model, vocab}, {data, model, vocab},...
  -- data is ordered batches X batchSize X smpLength
  local xE = torch.Tensor(self.batchSize, self.maxLength):type(TensorType)
  self.encoderModel:training()
  self.encoderModel:sequence()
  self.encoderModel:forget()
  self.encoderModel:setIterations(self.maxLength)
  local xDs = {}
  local targets = {}

  for num = 1, #self.decoderModels do
    self.decoderModels[num]:training()
    self.decoderModels[num]:sequence()
    self.decoderModels[num]:forget()
    self.decoderModels[num]:setIterations(self.maxLength)
    xDs[num] = torch.Tensor(self.batchSize , self.maxLength + 2):type(self.type)
  end
  local numBatches = math.floor(#self.inputData / self.batchSize)
  local numSamples = 1
  local lossVals = torch.FloatTensor(#self.decoderModels):zero()

  function getNextBatch(numData) --0 for encoded data, otherwise num of decoded data
    local vocab = self.encoderVocab
    local data = self.inputData
    local x = xE
    if numData > 0 then
      data = self.outputData[numData]
      x = xDs[numData]
      vocab = self.decoderVocabs[numData]
    end
    local padVal = vocab['<PAD>']
    local eosVal = vocab['<EOS>']
    local goVal = vocab['<GO>']
    if numSamples + self.batchSize > #data then
      return nil
    end
    x:fill(padVal)

    local currMaxLength = 0
    for i = 1, self.batchSize do
      local currSeq = data[i + numSamples - 1]
      local currLength = currSeq:nElement()
      currMaxLength = math.max(currMaxLength, currLength)
      if numData == 0 then --decoder
        x[{i,{1, currLength}}]:copy(currSeq)
      else
        x[{i,{2, currLength + 1}}]:copy(currSeq)
        x[{i, currLength + 2}] = eosVal
        x[{i, 1}] = goVal
      end
    end
    local target
    if numData == 0 then --decoder
      x = x:narrow(2, 1, currMaxLength)
    else
      target = x:narrow(2, 2, currMaxLength + 1):contiguous()
      x = x:narrow(2, 1, currMaxLength + 1)
    end
    return x, target
  end


  for b = 1, numBatches do
    local x = {}
    local yt = {}
    local y = {}
    local currLoss = 0
    x[0] = getNextBatch(0)
    self.encoderModel:zeroState()
    local out = self.encoderModel:forward(x[0])

    local state = self.encoderModel:getState()

    for m = 1, #decoderModels do
      self.decoderModels[m]:setState(state)
      x[m], yt[m] = getNextBatch(m)

      y[m] = self.decoderModels[m]:forward(x[m])
      currLoss = lossNoPadding(criterion, y[m], yt[m], self.decoderVocabs[m]['<PAD>'])
      --  print(torch.exp(currLoss))
      lossVals[m] = lossVals[m] + currLoss --/ opt.seqLength
      seqCriterion:forward(y[m],yt[m])

    end
    ----Training -> backpropagation
    if train then
      local f_eval = function()
        self.encoderModel:zeroGradParameters()
        self.encoderModel:zeroGradState()

        for m = 1, #decoderModels do
          self.decoderModels[m]:zeroGradParameters()
          self.decoderModels[m]:zeroGradState()

          local dE_dy = seqCriterion:backward(y[m],yt[m])
          self.decoderModels[m]:backward(x[m], dE_dy)
          local decoderGradState = self.decoderModels[m]:getGradState()
          self.encoderModel:accGradState(decoderGradState)
        end
        self.encoderModel:backward(x[0], out:zero())
        --Gradient clipping (actually normalizing)
        local norm = self.gradients:norm()
        if norm > self.optimConfig.gradClip then
          local shrink = self.optimConfig.gradClip / norm
          self.gradients:mul(shrink)
        end
        return currLoss, self.gradients
      end

      self.optimization(f_eval, self.weights, self.optimConfig)
    end
    numSamples = numSamples + self.batchSize

    xlua.progress(numSamples, #inputData)
  end

  xlua.progress(numSamples, #inputData)
  lossVals:div(numBatches)
  return lossVals:mean()
end
