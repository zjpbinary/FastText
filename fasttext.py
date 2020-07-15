import nltk
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
class Dataprepare:
    def __init__(self, trainfile, textfile, maxlen=100):
        super(Dataprepare, self).__init__()
        self.train = self.readfile(trainfile)
        self.test = self.readfile(textfile)
        self.worddic = self.add_dict(self.train)
        self.maxlen = maxlen
    def readfile(self, path):
        posfiles = os.listdir(path+'/pos')
        negfiles = os.listdir(path+'/neg')
        text = []
        for file in posfiles:
            with open(path+'/pos/'+file, 'r', encoding='utf-8') as f1:
                sent = nltk.word_tokenize(f1.read().strip())
                text.append((0, sent))
        for file in negfiles:
            with open(path+'/neg/'+file, 'r', encoding='utf-8') as f1:
                sent = nltk.word_tokenize(f1.read().strip())
                text.append((1, sent))
                random.shuffle(text)
        return list(zip(*text))
    def add_dict(self, pairs):
        worddic = {'<pad>': 0, '<unk>':1}
        i = 2
        for sent in pairs[1]:
            for word in sent:
                if word not in worddic:
                    worddic[word] = i
                    i += 1
        return worddic
    def padding(self, sent):
        l = len(sent)
        if l >= self.maxlen:
            return sent[:self.maxlen]
        else:
            return sent+[0 for _ in range(self.maxlen-l)]
    def indexed(self, pairs):
        return [pairs[0],
                [self.padding([self.worddic[word] if word in self.worddic else self.worddic['<unk>'] for word in sent]) for sent in pairs[1]]]
    def loadbatch(self, pairs, batchsize):
        i = 0
        batches = []
        while i+batchsize <= len(pairs[0]):
            batches.append([pairs[0][i:i+batchsize], pairs[1][i:i+batchsize]])
            i += batchsize
        return batches
    def run(self, batchsize):
        train = self.indexed(self.train)
        test = self.indexed(self.test)
        return self.loadbatch(train, batchsize), self.loadbatch(test, batchsize)

class FastText(nn.Module):
    def __init__(self, ntoken, nclass, embedsize, hiddensize):
        super(FastText, self).__init__()
        self.hiddensize = hiddensize
        self.embedsize = embedsize
        self.embedding = nn.Embedding(ntoken, self.embedsize, padding_idx=0)
        self.hiddenlayer = nn.Linear(self.embedsize, self.hiddensize)
        self.outlayer = nn.Linear(self.hiddensize, nclass)
    def forward(self, input):
        embedded = self.embedding(input)
        print(embedded.shape)
        hidden = self.hiddenlayer(embedded.mean(1))
        print(hidden.shape)
        out = self.outlayer(hidden)
        print(out.shape)
        return torch.softmax(out, 1)
    def starttrain(self, batches, model, device, criterion, optimizer, epoch):
        model.to(device)
        criterion.to(device)
        model.train()
        for i in range(epoch):
            for j, b in enumerate(batches):
                print(j)
                optimizer.zero_grad()
                predata = torch.LongTensor(b[1]).to(device)
                print(predata.shape)
                output = self.forward(predata)
                target = torch.LongTensor(b[0]).to(device)
                print(target.shape)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                print('第%d次迭代，'%i, '第%d个batch，'%j,'loss为：', loss)
    def eva(self, batches, model, device):
        model.to(device)
        model.eval()
        right = 0
        count = 0
        for batch in batches:
            input = torch.LongTensor(batch[1]).to(device)
            output = self.forward(input)
            predict = torch.argmax(output, 1)
            target = torch.LongTensor(batch[0]).to(device)
            count += len(target)
            right += torch.sum(torch.eq(predict, target))
        print('测试集上的精度为：', right.float()/count)
if __name__ == '__main__':
    loaddata = Dataprepare('data/train', 'data/test')
    trainbatches, testbatches = loaddata.run(100)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = FastText(len(loaddata.worddic), 2, 256, 256)
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    model.starttrain(trainbatches, model, device, criterion, optimizer, 1)
    model.eva(testbatches, model, device)




