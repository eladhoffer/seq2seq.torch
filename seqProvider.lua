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
  self.buffer:resize(numSeqs, self.maxLength):fill(self.padding)
  local startSeq = 1
  local addedLength = 0
  local currMaxLength = 0

  if self.startToken then
    self.buffer:select(2,1):fill(self.startToken)
    startSeq = 2
    addedLength = 1
  end
  if self.endToken then
    addedLength = addedLength + 1
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
