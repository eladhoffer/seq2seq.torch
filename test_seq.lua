require 'seqProvider_new'
p = seqProvider{source = '../../Datasets/training-monolingual/news-commentary-v6.en'}
b = p:getBatch(8)
