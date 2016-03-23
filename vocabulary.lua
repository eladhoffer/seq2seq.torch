local tds = require 'tds'

local function simpleTokenizer(sentence)
    local removedSymbols = '[\n,"?.!/\\<>():;]'
    local line = sentence:gsub(removedSymbols, '')
    line = line:gsub('%s+',' '):gsub('%s$',''):gsub("'", " '"):gsub("’", "’ "):lower()
    local words = line:split(' ')
    return words
end

local function createVocab(filename, tokenizer, vocab)
    local vocabSizeLimit = 2^31 - 1 --integer limit
    local vocabFreq = tds.Hash()
    local vocab = vocab or tds.Hash()
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
            if currWord then
              local encodedNum = vocab[currWord]
              if not encodedNum then
                  vocab[currWord] = currentNum
                  vocabFreq[currWord] = 0
                  currentNum = currentNum + 1
              end
              vocabFreq[currWord] = vocabFreq[currWord] + 1
            end
        end
    end
    file:close()
    return vocab, vocabFreq, currentNum - 1
end

local function sortFrequency(vocab, freq)
    local currentNum = 1
    for _ in pairs(vocab) do
        currentNum = currentNum + 1
    end
    local vocabFreq = torch.IntTensor(currentNum - 1):zero()
    for word,num in pairs(vocab) do
        vocabFreq[num] = freq[word]
    end
    local sortedIdxs
    vocabFreq, sortedIdxs = vocabFreq:sort(1, true)
    local newIdxs = torch.LongTensor(currentNum - 1)
    for i = 1, currentNum -1 do
        newIdxs[sortedIdxs[i]] = i
    end


    return vocab, vocabFreq
end



local vocabulary = torch.class('vocabulary')
function vocabulary:__init(...)
  local args = dok.unpack(
  {...},
  'vocabulary',
  'Sequence feeder',
    {arg='source', type='text', help='data source filename', required = true},
    {arg='tokenizer', type='function', help='tokenizing function', default=simpleTokenizer},
    {arg='unkToken', type='string', help='unknown token', default = '<UNK>'},
    {arg='padToken', type='string', help='unknown token', default = '<PAD>'},
    {arg='startToken', type='string', help='token at start of eache sequence', default = '<GO>'},
    {arg='endToken', type='string', help='token at start of eache sequence', default = '<EOS>'}
    )
    self.unkToken = args.unkToken
    self.tokenizer = args.tokenizer
    self.limitVocab = args.limitVocab
    self.startToken = args.startToken
    self.endToken = args.endToken
    self.padToken = args.padToken
    self.unkToken = args.unkToken

    local vocab, freq = createVocab(args.source, self.tokenizer, nil, true)
    self.vocab, self.vocabFreq = sortFrequency(vocab,freq)
    self:add(self.endToken, true)
    self:add(self.startToken, true)
    self:add(self.unkToken, true)
    self:add(self.padToken, true)
  end


function vocabulary:pad()
  return self.vocab[self.padToken]
end

function vocabulary:go()
  return self.vocab[self.startToken]
end

function vocabulary:unk()
  return self.vocab[self.unkToken]
end

function vocabulary:eos()
  return self.vocab[self.endToken]
end

function vocabulary:encode(str, minLength, maxLength)
  if str == nil then
    return nil
  end

  local minLength = minLength or 0
  local maxLength = maxLength or 2^31 -1

  local words = self.tokenizer(str)
  local length = #words
  if length < minLength then
    return  nil
  end
  length = math.min(length, maxLength)
  local encoded = torch.IntTensor(length):zero()

  for i=1, length do
    local currWord = words[i]
    local encodedNum = self.vocab[currWord]
    if not encodedNum then
      encodedNum = self:unk()
    end
    encoded[i] = encodedNum
  end
  return encoded
end

function vocabulary:createDecoder()
  self.decoder = tds.Hash()
  for word,num in pairs(self.vocab) do
    self.decoder[num] = word
  end
end

function vocabulary:decode(vec)
  if not self.decoder then
    self:createDecoder()
  end
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
    decoded[i] = self.decoder[vec[i]]
  end
  return decoded
end

function vocabulary:size()
  local vocabSize = 0
  for word,num in pairs(self.vocab) do
      vocabSize = vocabSize + 1
  end
  return vocabSize
end

function vocabulary:add(item, top)
  if item == nil or self.vocab[item] then return end
  if top then
    for word,num in pairs(self.vocab) do
      self.vocab[word] = num + 1
    end
    self.vocab[item] = 1
  else
    self.vocab[item] = self:size()+1
  end
  self.decoder = nil
end

function vocabulary:reduceFrequent(limit)
  if not limit then return end
  for word,num in pairs(self.vocab) do
    if num > limit then
      self.vocab[word] = nil
    end
  end
  self.decoder = nil
end

function vocabulary:saveAsText(filename)
  local file = io.open(filename, 'w')
  for w,n in pairs(self.vocab) do
    if n~=self:go() and n~=self:eos() and n~=self:unk() then
      file:write(w .. '\n')
    end
  end
  file:close()
end
