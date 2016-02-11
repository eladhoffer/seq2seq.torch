local function createDecodeFunc(vocab)
  local decoder = {}
  for word,num in pairs(vocab) do
    decoder[num] = word
  end
  local func = function(vec)
    if vec == nil then
      return nil
    end
    local length
    if torch.type(vec) == 'number' then
      vec = {vec}
    end
    if torch.isTensor(vec) then length = vec:size(1)
    else
      length = #vec
    end

    local decoded = {}
    for i=1, length do
      decoded[i] = decoder[vec[i]]
    end
    return decoded
  end
  return func
end

local function createEncodeFunc(vocab, tokenizer, buffer)
  local func = function(str, missingVal, minLength, maxLength)
    if str == nil then
      return nil
    end
    local minLength = minLength or 0
    local maxLength = maxLength or 2^31 -1
    local missingVal = missingVal or -1

    local words = tokenizer(str)
    local length = #words
    if length < minLength or length > maxLength then
      return nil
    end
    local encoded = buffer or torch.IntTensor(#words)
    encoded:zero()

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

local function createVocab(filename, tokenizer, vocab, getFreq)
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

function loadTextLinesVec(filename, vocab, encodeFunc, minLength, maxLength)
  local minLength = minLength or 1
  local maxLength = maxLength or 50
  local tds = require 'tds'
  local file = io.open(filename, 'r')
  local vector = tds.Vec()

  for line in file:lines() do
    local wordsVec = encodeFunc(line, -1, minLength, maxLength)
    if wordsVec then
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


local seqProvider = torch.class('seqProvider')
function seqProvider:__init(...)
  local args = dok.unpack(
  {...},
  'seqProvider',
  'Sequence feeder',
  {arg='source', type='text', help='data source filename', required = true},
  {arg='tokenizer', type='function', help='tokenizing function', default=simpleTokenizer},
    {arg='padding', type='number', help='padding value', default = 0},
    {arg='startToken', type='number', help='token at start of eache sequence (optional)'},
    {arg='endToken', type='number', help='token at start of eache sequence (optional)'},
    {arg='minLength', type='number', help='minimum sequence length', default = 3},
    {arg='maxLength', type='number', help='maximum sequence length', default = 50},
    {arg='type', type='string', help='type of output tensor', default = 'torch.ByteTensor'},
    {arg='cacheFolder', type='text', help='cache folder'},
    {arg='preprocess', type='boolean', help='save a preprocessed file', default = true}
    )
    self.padding = args.padding
    self.startToken = args.startToken
    self.endToken = args.endToken
    self.maxLength = args.maxLength
    self.minLength = args.minLength
    self.buffer = torch.Tensor():type(args.type)
    self.tokenizer = args.tokenizer
    self.preprocess = args.preprocess
    self.currentIndex = 1

    sys.execute('mkdir -p ' .. args.cacheFolder)

    local filename = args.source
    local cacheFilename = cacheFolder .. filename .. '_cached.t7'
    local vocabFilename = cacheFolder .. filename .. '_cached.t7'

    if paths.filep(vocabFilename) then
      self.vocab =torch.load(vocabFilename)
    else
      local vocab, freq = createVocab(args.source, self.tokenizer, {}, true)
      self.vocab = sortFrequency(vocab,freq)
    end

    self.encoder = createEncodeFunc(vocab, tokenizer)
    self.decoder = createDecodeFunc(vocab)

    if paths.filep(cacheFilename) then
      self.data = torch.load(cacheFilename)
    else
      if preprocess then
        self.data = loadTextLinesVec(args.source, self.vocab, self.encoder, self.minLength, self.maxLength)
        self.getItemFunc = function(x, idx) return x[idx] end
      else
        self.data = io.open(args.source, 'r')
        self.getItemFunc = function(x, idx)
          local line = x:read('*l')
          return self.encoder(line)
        end
      end
    end
  end

  function seqProvider:encode(str)
    return self.encoder(str)
  end

  function seqProvider:decode(vector)
    return self.decoder(vector)
  end

  function seqProvider:type(t)
    self.buffer = self.buffer:type(t)
    return self
  end

  function seqProvider:getIndexes(indexes)
    assert(self.preprocess, "getIndexes available only for preprocessed files")
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
      local currSeq = self.getItemFunc(self.data, indexes[i])
      local currLength = currSeq:nElement()
      currMaxLength = math.max(currMaxLength, currLength)
      self.buffer[i]:narrow(1, startSeq, currLength):copy(currSeq)
      if self.endToken then
        self.buffer[i][currLength + addedLength] = self.endToken
      end
    end
    return self.buffer:narrow(2, 1, currMaxLength + addedLength)
  end

  function seqProvider:getBatch(size)
    if self.currentIndex >= #self.data then
      return nil
    end
    local lastIndex = math.min(self.currentIndex + size - 1, #self.data)
    local range = torch.range(self.currentIndex, lastIndex)
    self.currentIndex = self.lastIndex
    return self:getIndexes(range)
  end

  function seqProvider:reset()
    self.currentIndex = 1
    if not self.preprocess then
      self.data:seek("set")
    end
  end
