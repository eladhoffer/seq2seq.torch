require 'nn'
require 'utils.OneHot'
require 'recurrent'
require 'eladtools'
require 'cunn'
local tds = require 'tds'
local TensorType = 'torch.CudaTensor'

 model = torch.load('./Results/PTB_LSTM_64_2/Net_3638.t7')
 data = torch.load('./cache/ptb_cached.t7')

local vocab = data.vocab
local decodeTbl = data.decodeTable

model.embedder = nn.OneHot(255, vocab['<PAD>'])
local enc = nn.Sequential():add(model.embedder):add(nn.Reverse(2)):add(model.recurrentEncoder):type(TensorType)
local dec = nn.Sequential():add(model.embedder):add(model.recurrentDecoder):add(model.classifier):add(nn.SoftMax()):type(TensorType)
enc:evaluate()
dec:evaluate()



function decode(x, num)
  local num = num or 50
  local predText = ''
  dec:single()
  local wordNum = torch.Tensor({vocab['<GO>']}):type(TensorType)
  dec:setState({x})
  for i=1, num do
      local pred = dec:forward(wordNum:view(1,-1))
      pred:select(2, vocab['<PAD>']):zero()
      _, wordNum = pred:max(2)
      wordNum = wordNum[1]
      if wordNum:squeeze() == vocab['<EOS>'] then
        break
      end
      predText = predText .. decodeTbl[wordNum:squeeze()]
  end
  return predText

end

function encode(s)
  local x = data.encodeFunc(s)
  local num = x:nElement()
  x = x:type(TensorType)
  local padToken = vocab['<PAD>']
  enc:zeroState()
  enc:sequence()
  enc:forward(x:view(1, -1))
  local state = enc:getState()
  return state[1]:clone()
end

function evalPTB(p)
local ptb = torch.load('ptb.t7')
local acc = 0
for i=1, #ptb do
  local decoded = decode(encode(ptb[i]))
  if decoded == ptb[i] then
    acc = acc + 1
  elseif p then
    print(decoded, ptb[i])
  end
end
print('PTB accuracy: ', acc / #ptb)
return acc / #ptb
end
