require 'torch'
require 'nn'
require 'optim'
require 'recurrent'
require 'eladtools'
require 'seqProvider'
require 'Seq2Seq'
-------------------------------------------------------

dofile('opt.lua')
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'LossRate.log')
local log = optim.Logger(logFilename)

----------------------------------------------------------------------
local encodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Language/books/books_large_p1.txt', maxLength = opt.seqLength, limitVocab = 20000, offsetStart = 1, offsetEnd = -1,
  type = opt.tensorType, preprocess = false, tokenizer = function(s) return s:lower():split(' ') end
}
local decodedSeq = seqProvider{
  source = '/home/ehoffer/Datasets/Language/books/books_large_p1.txt', maxLength = opt.seqLength, type = opt.tensorType, limitVocab = 20000,
  padStart = true, padEnd = true, preprocess = false, vocab = encodedSeq.vocab, offsetStart = 2, offsetEnd = 0
}
print('Dataset size (encoded=decoded)', encodedSeq:size(), decodedSeq:size())
if paths.filep(opt.load) then
    local modelConfig = torch.load(opt.load)
    print('==>Loaded Net from: ' .. opt.load)
    recurrentEncoder = modelConfig.recurrentEncoder
    recurrentDecoder = modelConfig.recurrentDecoder
    classifier = modelConfig.classifier
else
    recurrentEncoder = buildEncDec(encodedSeq.vocab:size(), true)
    recurrentDecoder = buildEncDec(decodedSeq.vocab:size(), false)
    classifier = nn.Linear(opt.rnnSize, decodedSeq.vocab:size())
end

function getEmbedder(m)
  return m:findModules('nn.LookupTable')[1]
end
getEmbedder(recurrentEncoder):share(getEmbedder(recurrentDecoder), 'weight','gradWeight')
--tie classification and embedding weights
classifier:share(recurrentDecoder:findModules('nn.LookupTable')[1], 'weight','gradWeight')

function zeroPadWeights(m, padIdx)
  local lookuptbl = getEmbedder(m)
  lookuptbl.paddingValue = padIdx
  lookuptbl.weight[padIdx]:zero()
end

zeroPadWeights(recurrentDecoder, decodedSeq.vocab:pad())
local criterion = nn.CrossEntropyCriterion():type(opt.tensorType)


local enc = recurrentEncoder
local dec = nn.Sequential()
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))


local criterion = nn.CrossEntropyCriterion()


local enc = nn.Sequential()
enc:add(recurrentEncoder)

local dec = nn.Sequential()
dec:add(recurrentDecoder)
dec:add(nn.TemporalModule(classifier))
print(enc)

-- Optimization configuration
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}

local seq2seq = Seq2Seq{
  encoder = enc,
  optimFunc =_G.optim[opt.optimization],
  optimState = optimState,
  criterion = criterion,
  batchSize = opt.batchSize
}
seq2seq:addDecoder(dec)
seq2seq:type(opt.tensorType)
seq2seq:flattenParameters()




----------------------------------------------------------------------
print '\n==> Encoder'
print(enc)

print '\n==> Decoder'
print(dec)
print('\n==>' .. seq2seq.params[1]:nElement() ..  ' Parameters')

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



------------------------------


function sample(encoderTbl, decoderTbl, str)
    local num = 50--opt.seqLength
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

    print('\nOriginal:\n' .. table.concat(encodeVocab:decode(encoded),' '))
    wordNum = torch.Tensor({decodeVocab:go()}):type(opt.tensorType)
    --if encoded:nElement() == 1 then
    --  encoded = encodeVocab:encode(str .. 'a'):type(opt.tensorType)
    --  encoded[encoded:size(1)]=0
    --end
--  if encoded:size(1) then encoderModel:sequence() end
    encoderModel:forward(encoded:view(1, -1))
    decoderModel:setState(encoderModel:getState())
    for i=1, num do
      --print(wordNum)
        local pred = decoderModel:forward(wordNum)
        pred:select(2, decodeVocab:unk()):zero()
        _, wordNum = pred:max(2)
        wordNum = wordNum[1]
        if wordNum:squeeze() == decodeVocab:eos() then
          break
        end
        table.insert(predText, decodeVocab:decode(wordNum)[1])
    end
    return table.concat(predText,' ')
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
    print('\nSampled Text:\n' .. sample({enc, encodedSeq.vocab}, {sampledDec, decodedSeq.vocab},'its a small quiet town , the kind where everyone knows your name .'))
    local LossTrain = seq2seq:learn(encodedSeq,{decodedSeq})
    saveModel(epoch)

    if opt.optState then
        torch.save(optStateFilename .. '_epoch_' .. epoch .. '.t7', optimState)
    end
    print('\nTraining Perplexity: ' .. torch.exp(LossTrain))

    epoch = epoch + 1


until false--stopTraining:update(LossVal)

local lowestLoss, bestIteration = stopTraining:lowest()

print("Best Iteration was " .. bestIteration .. ", With a validation loss of: " .. lowestLoss)
