require 'nn'
require 'optim'
require 'recurrent'
require 'MaskPadding'

local function lossNoPadding(criterion, y, yt, padToken)
    local loss = 0
    for i = 1, y:size(1) do
        local currLength = 1
        local currLoss = 0
        for j = 1, y:size(2) do
            if yt[i][j] == padToken then break end
            currLoss = currLoss + criterion:forward(y[i][j], yt[i][j])
            currLength = currLength + 1
        end
        loss = loss + currLoss / currLength
    end
    loss = loss / y:size(1)
    return loss
end


local Seq2Seq, parent = torch.class('Seq2Seq', 'nn.Container')
function Seq2Seq:__init(...)
  local args = dok.unpack(
  {...},
  'Seq2Seq',
  'Initializes a Seq2Seq ',
  {arg='encoder', type='table', help='encoder model'},
  {arg='maxLength', type='number', help='maximum sequence length', default = 50},
  {arg='batchSize', type='number', help='batch size used', default = 32},
  {arg='criterion', type='userdata', help='criterion used to evaluate', default = nn.CrossEntropyCriterion()},
  {arg='optimFunc', type='function', help='optimization function', default = optim.adam},
  {arg='optimState', type='table', help='optimization config', default = {}}
  )
  self.batchSize = args.batchSize
  self.optimFunc = args.optimFunc
  self.optimState = args.optimState
  self.encoderModel = args.encoder
  self.maxLength = args.maxLength
  self.decoderModels = {}
  self.criterion = args.criterion
  self.modules = {}
  parent.add(self, self.encoderModel)
end


function Seq2Seq:flattenParameters()
  self.params = self.params or {self:getParameters()}
end

function Seq2Seq:addDecoder(model)
  local num = #self.decoderModels + 1
  self.decoderModels[num] = model
  parent.add(self, model)
  return self
end


function Seq2Seq:encode(input)
  return self.encoderModel:forward(input)
end


function Seq2Seq:forward(inputEncoder, inputsDecoders)
  self.encoderModel:forward(inputEncoder)
  local state = self.encoderModel:getState()
  local output = {}
  for m = 1, #self.decoderModels do
    self.decoderModels[m]:setState(state)
    output[m] = self.decoderModels[m]:forward(inputsDecoders[m])
  end
  return output
end

function Seq2Seq:backward(inputEncoder, inputsDecoders, gradDecoders)
  local gradEncoder = self.encoderModel.output:zero()
  for m = 1, #self.decoderModels do
    self.decoderModels[m]:backward(inputsDecoders[m], gradDecoders[m])
    local decoderGradState = self.decoderModels[m]:getGradState()
    self.encoderModel:accGradState(decoderGradState)
  end
  return self.encoderModel:backward(inputEncoder, gradEncoder)
end

----------------------------------------------------------------------
function Seq2Seq:evaluate(encodedSeq, decodedSeqs, train)
    -- input is of form {data, model, vocab}, {data, model, vocab},...
    -- data is ordered batches X batchSize X smpLength
    local seqCriterion = nn.MaskPadding(self.criterion)
    local shuffle = false
    if train then
      shuffle = encodedSeq.preprocess
      self:training()
      self:flattenParameters()
    end
    local maxLength = self.maxLength
    self:sequence()
    self:forget()
    self:setIterations(maxLength)

    encodedSeq:reset()

    for m in pairs(decodedSeqs) do
      decodedSeqs[m]:reset()
      assert(decodedSeqs[m]:size() == encodedSeq:size())
    end
    local numBatches = math.floor((encodedSeq:size() -1)/ self.batchSize)
    local numSamples = 1
    local lossVals = torch.FloatTensor(#self.decoderModels):zero()
    local randIndexes = torch.LongTensor():randperm(encodedSeq:size())

    for b = 1, numBatches do
      local xEncoded
      local x = {}
      local yt = {}
      local currLoss = 0
      if shuffle then
       xEncoded = encodedSeq:getIndexes(randIndexes:narrow(1, numSamples, self.batchSize))
     else
       xEncoded = encodedSeq:getBatch(self.batchSize)--:getIndexes(randIndexes:narrow(1, numSamples, self.batchSize))
     end
      for m = 1, #self.decoderModels do
        local seq
        if shuffle then
          seq = decodedSeqs[m]:getIndexes(randIndexes:narrow(1, numSamples, self.batchSize))
        else
          seq = decodedSeqs[m]:getBatch(self.batchSize)
        end
        x[m] = seq:sub(1,-1,1,-2)
        yt[m] = seq:sub(1,-1,2,-1)
      end
      local y = self:forward(xEncoded, x)


      for m = 1, #self.decoderModels do
        currLoss = seqCriterion:forward(y[m],yt[m])
        lossVals[m] = lossVals[m] + currLoss --/ self.maxLength
      end

      ----Training -> backpropagation
      if train then
        local weights, gradients = unpack(self.params)
        local f_eval = function()
          self:zeroGradState()
          self:zeroGradParameters()
          local dE_dy = {}
          for m = 1, #self.decoderModels do
            dE_dy[m] = seqCriterion:backward(y[m],yt[m])
            dE_dy[m]:cmul(yt[m]:ne(0):view(yt[m]:size(1),yt[m]:size(2), 1):expandAs(dE_dy[m]))
            --print('dE')
          end
          self:backward(xEncoded, x, dE_dy)

          --Gradient clipping (actually normalizing)
          local norm = gradients:norm()
          if norm > 5 then
            local shrink = 5 / norm
            gradients:mul(shrink)
          end

          return currLoss, gradients
        end

        self.optimFunc(f_eval, weights, self.optimState)

      end
      numSamples = numSamples + self.batchSize

      xlua.progress(numSamples - 1, encodedSeq:size())
    end

    xlua.progress(encodedSeq:size(), encodedSeq:size())
    lossVals:div(numBatches)
    return lossVals[1]
  end
------------------------------
function Seq2Seq:learn(encodedSeq, decodedSeqs)
  return self:evaluate(encodedSeq, decodedSeqs, true)
end
