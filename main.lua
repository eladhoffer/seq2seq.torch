require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
require 'seqProvider'
require 'Seq2Seq'

dofile('opt.lua')

local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'LossRate.log')
----------------------------------------------------------------------
-- Model + Loss:
local encodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.en', maxLength = opt.seqLength, limitVocab = 50000,
  type = opt.tensorType, preprocess = false
}
local decodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Language/news_commentary_v10/news-commentary-v10.fr-en.fr', maxLength = opt.seqLength, limitVocab = 50000,
  type = opt.tensorType, padStart = true, padEnd = true, preprocess = false
}

--print(decodedSeq:decode(decodedSeq:getBatch(8)[1]))
print('Encoded Vocabulary:', encodedSeq.vocab:size())
print('Decoded Vocabulary:', decodedSeq.vocab:size())
print('Dataset size (encoded == decoded)', encodedSeq:size(), decodedSeq:size())

if paths.filep(opt.load) then
    local modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
    recurrentEncoder = modelConfig.recurrentEncoder
    recurrentDecoder = modelConfig.recurrentDecoder
    classifier = modelConfig.classifier
else
    recurrentEncoder = buildRNN(encodedSeq.vocab:size(), true)
    recurrentDecoder = buildRNN(decodedSeq.vocab:size(), false)
    classifier = nn.Linear(opt.rnnSize, decodedSeq.vocab:size())
end

--tie classification and embedding weights
classifier:share(recurrentDecoder:findModules('nn.LookupTable')[1], 'weight','gradWeight')

function zeroPadWeights(m, padIdx)
  local lookuptbl = m:findModules('nn.LookupTable')[1]
  lookuptbl.paddingValue = padIdx
  lookuptbl.weight[padIdx]:zero()
end

zeroPadWeights(recurrentEncoder, encodedSeq.vocab:pad())
zeroPadWeights(recurrentDecoder, decodedSeq.vocab:pad())
local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)


local enc = recurrentEncoder
local dec = nn.Sequential()
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))


--sequential criterion
local seqCriterion = nn.TemporalCriterion(criterion)

-- Optimization configuration
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}



local seq2seq = Seq2Seq{
  encoder = enc,
  optimFunc=_G.optim[opt.optimization],
  optimState = optimState,
  criterion = criterion,
  batchSize = opt.batchSize,
  padValue = decodedSeq.vocab:pad()
}
seq2seq:addDecoder(dec)
seq2seq:type(opt.tensorType)
seq2seq:flattenParameters()

----------------------------------------------------------------------
print '\n==> Encoder'
print(enc)

print '\n==> Decoder'
print(dec)

print '\n==> Criterion'
print(criterion)


----------------------------------------------------------------------
---utility functions
function saveModel(epoch)
  local fn = netFilename .. '_' .. epoch .. '.t7'
  torch.save(fn,
  {
    recurrentEncoder = recurrentEncoder:clearState(),
    recurrentDecoder = recurrentDecoder:clearState(),
    classifier = classifier:clearState(),
    vocabEncode = encodedSeq.vocab,
    vocabDecode = decodedSeq.vocab
  })
  collectgarbage()
end



function sample(encoderTbl, decoderTbl, str)
    local num = #str:split(' ')
    local encoderModel, encodeVocab = unpack(encoderTbl)
    local decoderModel, decodeVocab = unpack(decoderTbl)
    encoderModel:zeroState()
    decoderModel:single()
    encoderModel:sequence()
    encoderModel:evaluate()
    decoderModel:evaluate()
    local pred, predText, embedded
    predText = {}
    local encoded = encodeVocab:encode(str):type(opt.tensorType)
    print('\nOriginal:\n' .. table.concat(encodeVocab:decode(encoded) ,' '))
    wordNum = torch.Tensor({decodeVocab:go()}):type(opt.tensorType)
    if encoded:nElement() == 1 then
      encoderModel:single()
    end
    encoderModel:forward(encoded:view(1, -1))
    decoderModel:setState(encoderModel:getState())
    for i=1, num do
        local pred = decoderModel:forward(wordNum)
        pred:select(2, decodeVocab:unk()):zero()
        _, wordNum = pred:max(2)
        wordNum = wordNum[1]
        if wordNum:squeeze() == decodeVocab:eos() then
          break
        end
        table.insert(predText, decodeVocab:decode(wordNum)[1])
    end
    return table.concat(predText, ' ')
end


local sampledDec = nn.Sequential():add(recurrentDecoder):add(classifier):add(nn.SoftMax():type(opt.tensorType))

local decreaseLR = EarlyStop(1,opt.epochDecay)
local stopTraining = EarlyStop(opt.earlyStop, opt.epoch)
local epoch = 1



repeat

    print('\nEpoch ' .. epoch ..'\n')
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'Lately, with gold prices up more than 300% over the last decade, it is harder than ever.'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'a form of asbestos once used to make kent cigarette filters has caused a high percentage of cancer deaths among a group of workers'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},"the following were among yesterday 's offerings and pricings in the u.s. and non-u.s. capital markets"))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'nothing could be further from the truth'))
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'Are you one of millions out there who are trying to learn foreign language, but never have enough time?'))
    --local LossTrain = seq2seq({enc, encodedSeq},{{dec, decodedSeq}}, true)
      local LossTrain = seq2seq:learn(encodedSeq,{decodedSeq})
    --local LossTrain = seq2seq({enc, encodedSeq},{{dec1, decodedSeq}, {dec2, decodedSeq}}, true)
    saveModel(epoch)

    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    --local LossVal = evaluate(data.validationData)

    --print('\nValidation Perplexity: ' .. torch.exp(LossVal))

    --local LossTest = evaluate(data.testData)


    --print('\nTest Perplexity: ' .. torch.exp(LossTest))
    --log:add{['Training Loss']= LossTrain, ['Validation Loss'] = LossVal, ['Test Loss'] = LossTest}
    --log:style{['Training Loss'] = '-', ['Validation Loss'] = '-', ['Test Loss'] = '-'}
    --log:plot()
    epoch = epoch + 1

    --if decreaseLR:update(LossVal) then
    --    optimState.learningRate = optimState.learningRate / opt.decayRate
    --    print("Learning Rate decreased to: " .. optimState.learningRate)
    --    decreaseLR = EarlyStop(1,1)
    --    decreaseLR:reset()
    --end

until false--stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
