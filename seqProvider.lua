local tds = require 'tds'
local tokens = {'<UNK>', '<EOS>', '<GO>', '<PAD>'}



local function decodeFunc(decoder, seperator)
  local seperator = seperator or ' '
  local func = function(vec)
    local output = ''
    for i=1, vec:size(1) do
      local decoded = decoder[vec[i]] or '<UNK>'
      output = output .. seperator .. decoded
    end
    return output
  end
  return func
end

local function encodeFunc(vocab, missingVal)
  local missingVal = missingVal or -1
  local func = function(str)
    local words = str:split(' ')
    local length = #words
    local encoded = torch.IntTensor(#words):zero()

    for i=1, length do
      local currWord = words[i]
      local encodedNum = vocab[currWord]
      if not encodedNum then
        encodedNum = missingVal
      end
      encoded[i] = encodedNum
    end
    return encoded
  end

  return func
end

local function simpleTokenizer(sentence)
  local removedSymbols = '[\n,"?.!/\\<>():;]'
  local line = sentence:gsub(removedSymbols, ' ')
  line = line:gsub('%s+',' '):gsub('%s$',''):gsub("'s", " 's"):lower()
  local words = line:split(' ')
  return words
end

local function createDictionary(filename, tokenizer, vocab, getFreq)
  local vocabSizeLimit = 2^31 - 1 --integer limit
  local vocabFreq
  if getFreq then
    vocabFreq = {}
  end
  local vocab = vocab or {}
  local currentNum = 1
  for _ in pairs(vocab) do
    currentNum = currentNum + 1
  end

  local file = io.open(filename, 'r')

  for line in file:lines() do
    local words = tokenizer(line)
    local length = #words
    for i=1, length do
      local currWord = words[i]
      local encodedNum = vocab[currWord]
      if not encodedNum then
        vocab[currWord] = currentNum
        if getFreq then
          vocabFreq[currWord] = 0
        end
        currentNum = currentNum + 1
      end
      if getFreq then
        vocabFreq[currWord] = vocabFreq[currWord] + 1
      end
    end
  end
  return vocab, vocabFreq, currentNum - 1
end

local function sortFrequency(vocab, freq)
  local currentNum = 1
  for _ in pairs(vocab) do
    currentNum = currentNum + 1
  end
  local vocabFreq = torch.IntTensor(currentNum - 1):zero()
  for word,num in pairs(vocab) do
    vocabFreq = freq[words]
  end
  local sortedIdxs
  vocabFreq, sortedIdxs = vocabFreq:sort(1, true)
  local newIdxs = torch.LongTensor(currentNum - 1)
  for i = 1, currentNum -1 do
    newIdxs[sortedIdxs[i]] = i
  end

  local decodeTable = {}
  for word, num in pairs(vocab) do
    vocab[word] = newIdxs[num]
    decodeTable[newIdxs[num]] = word
  end
  return vocab, decodeTable
end

function loadTextLinesVec(filename, vocab, tokenizer, minLength, maxLength)
  local minLength = minLength or 1
  local maxLength = maxLength or 50
  local tds = require 'tds'
  local file = io.open(filename, 'r')
  local vector = tds.Vec()

  for line in file:lines() do

    local words = tokenizer(line)
    local length = #words
    if length > minLength and length < maxLength then
      local wordsVec = torch.IntTensor(#words):zero()
      for i=1, length do
        wordsVec[i] = vocab[currWord]
      end
      vector:insert(wordsVec)
    end
  end

  return vector
end

function reduceNumWords(sentences, num, replaceWith)
  for _, v in pairs(sentences) do
    for i=1, v:size(1) do
      if v[i] > num then
        v[i] = replaceWith
      end
    end
  end
  return sentences
end


local dataFolder = '../../Datasets/training-monolingual/'--'../../Datasets/Books/'--
local cacheFolder = './cache/'
sys.execute('mkdir -p ' .. cacheFolder)

local filename = 'news-commentary-v6.en'--'europarl-v6.en'
local cacheFilename = cacheFolder .. filename .. '_cached.t7'

local data
if paths.filep(cacheFilename) then
  data = torch.load(cacheFilename)
else
  data = loadTextLines(dataFolder .. filename ,tokens)
  torch.save(cacheFilename, data)
end
return data


local seqProvider = torch.class('seqProvider')
function seqProvider:__init(...)
  local args = dok.unpack(
  {...},
  'seqProvider',
  'Sequence feeder',
  {arg='data', type='userdata', help='data source', required = true},
  {arg='padding', type='number', help='padding value', default = 0},
  {arg='startToken', type='number', help='token at start of eache sequence (optional)'},
  {arg='endToken', type='number', help='token at start of eache sequence (optional)'},
  {arg='maxLength', type='number', help='maximum sequence length', default = 50},
  {arg='type', type='string', help='type of output tensor', default = 'torch.ByteTensor'}
  )
  self.padding = args.padding
  self.startToken = args.startToken
  self.endToken = args.endToken
  self.data = args.data
  self.maxLength = args.maxLength
  self.buffer = torch.Tensor():type(args.type)
end

function seqProvider:type(t)
  self.buffer = self.buffer:type(t)
  return self
end

function seqProvider:getSequences(indexes)
  local numSeqs = indexes:size(1)
  local startSeq = 1
  local addedLength = 0
  local currMaxLength = 0

  if self.startToken then
    startSeq = 2
    addedLength = 1
  end
  if self.endToken then
    addedLength = addedLength + 1
  end

  local bufferLength = self.maxLength + addedLength
  self.buffer:resize(numSeqs, bufferLength):fill(self.padding)
  if self.startToken then
    self.buffer:select(2,1):fill(self.startToken)
  end

  for i = 1, numSeqs do
    local currSeq = self.data[indexes[i]]
    local currLength = currSeq:nElement()
    currMaxLength = math.max(currMaxLength, currLength)
    self.buffer[i]:narrow(1, startSeq, currLength):copy(currSeq)
    if self.endToken then
      self.buffer[i][currLength + addedLength] = self.endToken
    end
  end
  return self.buffer:narrow(2, 1, currMaxLength + addedLength)
end
