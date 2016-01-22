local function getParametersMultiple(...)
  local tmpModel = nn.Sequential()
  for m in pairs({...}) do
    tmpModel:add(m)
  end
  return tmpModel:getParameters()
end


function seq2seq(encoderTbl, decoderTbls, train)
  local args = dok.unpack(
  {...},
  'seq2seq',
  'Train or evaluate a seq2seq ',
  {arg='encoder', type='table', help='encoder - {model, data, vocab}'},
  {arg='decoders', type='table', help='decoders model - {{model1, data1, vocab1}, {model2,data2, vocab2},...}'},
  {arg='maxLength', type='number', help='maximum sequence length', default = 50},
  {arg='batchSize', type='number', help='batch size', default = 128},
  {arg='train', type='boolean', help='train the network or evaluate only'},
  {arg='tensorType', type='string', help='type of tensor to feed throught the network', default='torch.CudaTensor'},
  {arg='trainingConfig', type='table', help='table with optimization params + optionally weight and grads',
   default={
     optimFunc = optim.adam,
     learningRate = 1e-3,
     gradClip = 5
   }
 })
  -- input is of form {data, model, vocab}, {data, model, vocab},...
  -- data is ordered batches X batchSize X smpLength
  local maxLength = args.maxLength
  local batchSize = args.batchSize
  local train = args.training

  local inputData, encoderModel, encoderVocab = unpack(args.encoder)
  local xE = torch.Tensor(batchSize, maxLength):type(args.tensorType)
  if train then
    encoderModel:training()
    if (not args.trainingConfig.weights) or (not args.trainingConfig.gradients) then
      args.trainingConfig.weights, args.trainingConfig.gradients = getParametersMultiple(encoderModel, unpack(decoderModels))
    end
    args.trainingConfig.optimFunc = args.trainingConfig.optimFunc or optim.adam
    args.trainingConfig.gradClip = args.trainingConfig.gradClip or 5
  else
    encoderModel:evaluate()
  end
  encoderModel:sequence()
  encoderModel:forget()
  encoderModel:setIterations(maxLength)
  local outputData = {}
  local decoderModels = {}
  local decoderVocabs = {}
  local xDs = {}
  local targets = {}

  for num, decTbl in pairs(decoderTbls) do
    outputData[num], decoderModels[num], decoderVocabs[num] = unpack(args.decoders)
    if train then
      decoderModels[num]:training()
    else
      decoderModels[num]:evaluate()
    end
    decoderModels[num]:sequence()
    decoderModels[num]:forget()
    decoderModels[num]:setIterations(maxLength)
    xDs[num] = torch.Tensor(batchSize , maxLength + 2):type(args.tensorType)
  end
  local numBatches = math.floor(#inputData / batchSize)
  local numSamples = 1
  local lossVals = torch.FloatTensor(#decoderModels):zero()

  function getNextBatch(numData) --0 for encoded data, otherwise num of decoded data
    local vocab = encoderVocab
    local data = inputData
    local x = xE
    if numData > 0 then
      data = outputData[numData]
      x = xDs[numData]
      vocab = decoderVocabs[numData]
    end
    local padVal = vocab['<PAD>']
    local eosVal = vocab['<EOS>']
    local goVal = vocab['<GO>']
    if numSamples + batchSize > #data then
      return nil
    end
    x:fill(padVal)

    local currMaxLength = 0
    for i = 1, batchSize do
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
    encoderModel:zeroState()
    local out = encoderModel:forward(x[0])

    local state = encoderModel:getState()

    for m = 1, #decoderModels do
      decoderModels[m]:setState(state)
      x[m], yt[m] = getNextBatch(m)

      y[m] = decoderModels[m]:forward(x[m])
      currLoss = lossNoPadding(criterion, y[m], yt[m], decoderVocabs[m]['<PAD>'])
    --  print(torch.exp(currLoss))
      lossVals[m] = lossVals[m] + currLoss --/ opt.seqLength
      seqCriterion:forward(y[m],yt[m])

    end
    ----Training -> backpropagation
    if train then
      local weights = args.trainingConfig.weights
      local gradients = args.trainingConfig.gradients

      local f_eval = function()
        encoderModel:zeroGradParameters()
        encoderModel:zeroGradState()

        for m = 1, #decoderModels do
          decoderModels[m]:zeroGradParameters()
          decoderModels[m]:zeroGradState()

          local dE_dy = seqCriterion:backward(y[m],yt[m])
          decoderModels[m]:backward(x[m], dE_dy)
          local decoderGradState = decoderModels[m]:getGradState()
          encoderModel:accGradState(decoderGradState)
        end
        encoderModel:backward(x[0], out:zero())
        --Gradient clipping (actually normalizing)
        local norm = gradients:norm()
        if norm > trainingConfig.gradClip then
          local shrink = trainingConfig.gradClip / norm
          gradients:mul(shrink)
        end
        return currLoss, gradients
      end

      args.trainingConfig.optimFunc(f_eval, weights, args.trainingConfig)
    end
    numSamples = numSamples + batchSize

    xlua.progress(numSamples - 1, #inputData)
  end

  xlua.progress(#inputData, #inputData)
  lossVals:div(numBatches)
  return lossVals:squeeze()
end
