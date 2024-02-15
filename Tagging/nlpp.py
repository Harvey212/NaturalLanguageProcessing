# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertForTokenClassification
from transformers import BertConfig
from transformers import AdamW
import pandas as pd
import openpyxl
import numpy as np
from torch.autograd import Variable


def strQ2B(s):
	rstring=""
	for uchar in s:
		u_code=ord(uchar)
		if u_code==12288:
			u_code=32
		elif 65281<=u_code<=65374:
			u_code-=65248
		
		rstring+=chr(u_code)

	return rstring


def clear(TEX):

	TEX=strQ2B(TEX)
	TEX=TEX.lower()
	TEX=TEX.replace('…','.')
	TEX=TEX.replace('誒','耶')
	TEX=TEX.replace('痾','啊')
	TEX=TEX.replace('厠','廁')
	TEX=TEX.replace('擤','弄')
	TEX=TEX.replace('搥','捶')
	TEX=TEX.replace('嵗','歲')
	TEX=TEX.replace('曡','疊')
	TEX=TEX.replace('厰','廠')
	TEX=TEX.replace('聼','聽')
	TEX=TEX.replace('柺','拐')

	return TEX






class paragraph:
	def  __init__(self,idd):
		self.article_id=idd
		self.start_position=[]
		self.end_position=[]
		self.entity_text=[]
		self.entity_type=[]
		self.para=[]

		self.sentences=[]
		self.endindex=[]
		self.labels=[]
		self.features=[]

		self.tagtonum=dict()

		#no tag->4800

		self.tagtonum['name']=1     #7
		self.tagtonum['location']=2 #7 
		self.tagtonum['time']=3     #70
		self.tagtonum['contact']=4  #1
		self.tagtonum['ID']=5       #0.5
		self.tagtonum['profession']=6#1
		self.tagtonum['family']=7    #1
		self.tagtonum['clinical_event']=8 #0.1
		self.tagtonum['organization']=9  #0.1
		self.tagtonum['education']=10  #0.1
		self.tagtonum['money']=11   #3
		self.tagtonum['med_exam']=12 #9
		self.tagtonum['others']=13 #0.1


	def add_start_position(self,start):
		self.start_position.append(start)
	def add_end_position(self,end):
		self.end_position.append(end)
	def add_entity_text(self,word):
		self.entity_text.append(word)
	def add_entity_type(self,typp):
		self.entity_type.append(typp)
	def add_para(self,para):
		self.para.append(para)

	def get_id(self):
		return self.article_id
	def get_start_position(self):
		return self.start_position
	def get_end_position(self):
		return self.end_position
	def get_entity_text(self):
		return self.entity_text
	def get_entity_type(self):
		return self.entity_type
	def get_para(self):
		return (self.para)[0]

	def sepsentence(self):
		
		text=self.get_para()
		sentences=[]
		sentence=''
		for i in text:
			if i!='？' and i!='。' and i!='～':  # and i!='，'
				sentence+=i
			else:
				sentence+=i
				if len(sentence)<400:
					sentences.append(sentence)
				else:
					fakesent=sentence.split('，')
					for h in range(len(fakesent)-1):
						if h%2==0:
							accs=[]
							accs.append(fakesent[h])
						else:
							accs.append(fakesent[h])
							ad='，'.join(accs)
							ad+='，'
							sentences.append(ad)
					lastsent=fakesent[len(fakesent)-1]

					if len(accs)==1:
						tem=accs[0]+'，'+lastsent
						sentences.append(tem)
					else:
						sentences.append(lastsent)

				sentence=''

		#####################################
		##separate based on speaking person 
		#finalsentences=[]

		#startcount=0

		#for k in sentences:
		#	if startcount==0:
		#		tempsentence=k
		#		startcount+=1
		#	else:
		#		if '：' in k:
		#			finalsentences.append(tempsentence)
		#			tempsentence=''
		#			tempsentence+=k
		#		else:
		#			tempsentence+=k

		#finalsentences.append(tempsentence)

		#self.sentences=finalsentences
		################################################
		self.sentences=sentences

	def createSentEndIndex(self):

		sents=self.sentences
		endindex=[]
		acc=-1
		for k in sents:
			acc=acc+(len(k))
			endindex.append(acc)

		self.endindex=endindex

	def createLabels(self):

		self.sepsentence()
		self.createSentEndIndex()

		targetstart=self.get_start_position()
		targetend=self.get_end_position()
		tags=self.get_entity_type()
		
		labels=[]
		features=[]
		pth=0
		sentstart=0

		tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


		for i in range(len(self.sentences)):

			tempsent=clear(self.sentences[i])
			com=tokenizer.tokenize(tempsent)

			##
			if '[UNK]' in com:
				print('hello')
				print(com)
				print(self.sentences[i])
			##


			feature=tokenizer.encode(tempsent, return_tensors="pt")
			features.append(feature)

			ans=[]

			for m in com:
				rr=m.replace('#','')
				ans.append(rr)

			anscount=-1
			ansend=[]  #relative

			for d in range(len(ans)):
				anscount+=len(ans[d])
				ansend.append(anscount)


			label=[0]*len(ans)
			sentend=self.endindex[i]   #absolute end of sententence
			seeNext=True

			while seeNext:
				if pth<len(targetend):
					if ((int(targetend[pth])-1)<=sentend):
						relstart=int(targetstart[pth])-sentstart
						relend=(int(targetend[pth])-1)-sentstart
						tagg=tags[pth]
						lab=self.tagtonum[tagg]


						stk=0
						enk=0

						limit=len(ans)

						see1=True

						while see1:
							if stk<limit:
								if ansend[stk]<relstart:  ##relstart=3 ansend[stk]=4  stk=3
									stk+=1
								else:
									see1=False
							else:
								see1=False


						see2=True

						while see2:
							if enk<limit:
								if ansend[enk]<relend:  #relend=10 ansend[enk]=10 enk=5
									enk+=1
								else:
									see2=False
							else:
								see2=False


						###############
						##example
						#民', '眾', 'line','嗨'
						#relstart=2  ansend[stk]=5 stk=2
						#relend5=5   ansend[enk]   enk=2 
						##############

						if stk==enk:                 
							label[stk]=lab
						else:
							for h in range(stk,(enk+1)):
								label[h]=lab
					
						pth+=1
					else:
						seeNext=False
				else:
					seeNext=False
			

			sentstart=sentend+1
			label.insert(0,0)  #CLS
			label.append(0)    #SEP
			labels.append(label)
			
		self.labels=labels
		self.features=features
		#############################################
		##to know which sentence the word belongs to
		#whichsentence=[]

		#for i in range(len(targetwordpositions)):
		#	wh=int(targetwordpositions[i])

		#	thcount=0
		#	end=self.endindex[thcount]

		#	while wh>end:
		#		thcount+=1
		#		end=self.endindex[thcount]
	
		#	targetsent=self.sentences[thcount]
		#	whichsentence.append(targetsent)

		#self.whichsentences=whichsentence
		#self.tagtonum

		#return self.whichsentences
		##########################################
	def getlabels(self):
		return self.labels
	def getsentences(self):
		return self.sentences
	def getfeatures(self):
		return self.features

corpus=[]

f = open("train_2.txt", "r", encoding="utf8")
lines = f.readlines()

start=0
resflag=False
paracount=0

inputcount=False

for line in lines:
	if start==0:
		start+=1
		newpara=paragraph(paracount)
		newpara.add_para(line.replace("\n",''))
		paracount+=1

	else:
		if line=='\n':
			resflag=False

		if 'article_id	start_position	end_position	entity_text	entity_type' in line:
			resflag=True
		if resflag==True and ('article_id	start_position	end_position	entity_text	entity_type' not in line):
			see=line.replace("\n",'')
			res=see.split("\t")
			newpara.add_start_position(res[1])
			newpara.add_end_position(res[2])
			newpara.add_entity_text(res[3])
			newpara.add_entity_type(res[4])
		
		if '--------------------' in  line:
			corpus.append(newpara)
			inputcount=True

		if inputcount==True and line!='\n' and ('--------------------' not in  line):
			newpara=paragraph(paracount)
			newpara.add_para(line.replace("\n",''))
			paracount+=1
			inputcount=False


class proj:
	def __init__(self,cor):
		self.corpus=cor

		self.numlabels=14
		self.tagtonum=dict()

		#no tag->4800

		self.tagtonum[1]='name'     #7
		self.tagtonum[2]='location' #7 
		self.tagtonum[3]='time'     #70
		self.tagtonum[4]='contact'  #1
		self.tagtonum[5]='ID'       #0.5
		self.tagtonum[6]='profession'#1
		self.tagtonum[7]='family'    #1
		self.tagtonum[8]='clinical_event' #0.1
		self.tagtonum[9]='organization'  #0.1
		self.tagtonum[10]='education'  #0.1
		self.tagtonum[11]='money'  #3
		self.tagtonum[12]='med_exam'#9
		self.tagtonum[13]='others'#0.1

		#
		#self.tagtonum[7]='biomarker'
		#self.tagtonum[10]='special_skills'
		#self.tagtonum[11]='unique_treatment'
		#self.tagtonum[12]='account'
		#self.tagtonum[16]='belonging_mark'
		#

		self.model=BertForTokenClassification.from_pretrained('bert-base-chinese',num_labels=self.numlabels)
		self.tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')

		self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
		self.finalres=[]
		self.weight=[]

	def train(self):
		self.model.train()
		epoch=0
		#self.setweight()

		for i in range(len(self.corpus)):
			currcorpus=self.corpus[i]
			currcorpus.createLabels()
			xs=currcorpus.getfeatures()
			ys=currcorpus.getlabels()

			for g in range(len(xs)):
				fea=xs[g]
				lala=ys[g]

				####################
				#mask=[1]*len(lala)
				#mask[0]=0
				#mask[len(lala)-1]=0
				#,attention_mask=torch.tensor(mask).unsqueeze(0)
				##########################
				
				###########################
				#positions=[]
				#pos=0
				#for z in range(len(lala)):
				#	if z==0:
				#		prev=lala[z]
				#		positions.append(pos)
				#	if z==1:
				#		curr=lala[z]
				#		if curr==0:
				#			pos+=1
				#			positions.append(pos)

				#	if z>1:
				#		prev=curr
				#		curr=lala[z]
				######################################

				#if sum(lala)!=0:
				outputs=self.model.forward(input_ids=fea, labels=torch.tensor(lala).unsqueeze(0))

				#
				#LOO = outputs.logits
				#fakepred = torch.argmax(LOO, dim=2) #include cls, sep
				#loss=self.calLoss(fakepred,lala)
				#
				loss = outputs.loss
				#
				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()

				print(epoch)
				epoch+=1

		self.model.save_pretrained('C:/Users/chrystal212/Desktop/nlpp')

	def calLoss(self,inp,tarr):
		weight=[]

		for i in range(len(tarr)):
			weight.append(self.weight[tarr[i]])

		tarr=torch.tensor(tarr)

		weight=torch.tensor(weight)
		weight=weight.float()
		lolo=torch.nn.MultiLabelSoftMarginLoss(weight=weight)(inp.float(),tarr.float())
		fll = Variable(lolo, requires_grad=True)

		return fll



	def load_pretrain(self):
		config = BertConfig.from_json_file('C:/Users/chrystal212/Desktop/nlpp/config.json')
		path='C:/Users/chrystal212/Desktop/nlpp/'
		self.model=BertForTokenClassification.from_pretrained(pretrained_model_name_or_path=path,config=config)

		##
		#self.model.eval()
		#text='民眾：阿只是前天好很多。前天就算沒盜，'
		#temp=clear(text)
		#see=self.tokenizer.tokenize(temp)
		#inputs = self.tokenizer.encode(temp, return_tensors="pt")
		#res=self.model(inputs)
		#outputs = res.logits
		#pred = torch.argmax(outputs, dim=2)
		#print(see)
		#print(pred)
		##

	def evaluate(self,dat):
		self.model.eval()

		for k in range(len(dat)):
			td=dat[k]
			td.sepsentence()
			sents=td.getsentences()
			paraindex=0

			for j in range(len(sents)):
				text=sents[j]
				temp=clear(text)

				inputs = self.tokenizer.encode(temp, return_tensors="pt")  #include cls ,sep
				ref=self.tokenizer.tokenize(temp)  #no cls ,sep   #real character , may include #

				#
				if '[UNK]' in ref:
					print('yes')
					print(ref)
					print(text)
				#誒  耶
				#痾  啊
				#厠  廁
				#擤  弄


				with torch.no_grad():
					res=self.model(inputs)   
					outputs = res.logits
					pred = torch.argmax(outputs, dim=2) #include cls, sep

					pred=(pred.tolist())[0]
					
					mark=False
					word=''
					startindex=0 #character based
					endindex=0  #character based

					flag=False
					flag2=False

					for t in range(len(pred)):
						if t!=0 and t!=(len(pred)-1):
							tar=(ref[t-1]).replace('#','')
							endindex+=len(tar)  #nextstart
							if pred[t]==0:
								if mark==True:
									flag=True
									mark=False
								else:
									startindex=endindex

							else:
								if mark==False:
									tagg=self.tagtonum[pred[t]]  #tagg is string
									word+=tar
									mark=True
								else:
									temptag=self.tagtonum[pred[t]]
									if temptag==tagg:
										word+=tar
									else:
										flag=True
										flag2=True

							if flag==True:
								#print('yes')
								wordlen=len(word)
								wordend=startindex+wordlen
								

								#try:
								realword=''
								for p in range(startindex,wordend):   #relative 
									realword+=text[p]

								realstart=paraindex+startindex
								realend=paraindex+wordend
								td.add_start_position(realstart)
								td.add_end_position(realend)
								td.add_entity_text(realword)
								td.add_entity_type(tagg)
								#except:
								#print(text)
								#print(startindex)
								#print(wordend)
								#print(pred)
								#print(ref)
								#print(realword)
								

								
								flag=False
								word=''

								if flag2==True:
									tagg=self.tagtonum[pred[t]]
									word+=tar
									mark=True
									flag2=False
									startindex=wordend
								else:
									startindex=endindex


				paraindex+=len(text)

			print(k)
			self.finalres.append(td)

		self.generateresult()

	def generateresult(self):
		finalframe=[]

		for i in range(len(self.finalres)):
			currpara=self.finalres[i]
			currid=currpara.get_id()
			parastarts=currpara.get_start_position()
			paraends=currpara.get_end_position()
			paraEntityText=currpara.get_entity_text()
			paraEntityType=currpara.get_entity_type()
			for j in range(len(paraEntityText)):
				temp=[]
				temp.append(currid)
				temp.append(parastarts[j])
				temp.append(paraends[j])
				temp.append(paraEntityText[j])
				temp.append(paraEntityType[j])

				finalframe.append(temp)

		df = pd.DataFrame(finalframe, columns=['article_id', 'start_position', 'end_position','entity_text','entity_type'])
	
		df.to_excel('nlpproject.xlsx',index=False)

	def setweight(self):
		
		#counts=[4000, 7, 7, 70, 1, 0.5, 1, 1, 0.1, 0.1, 0.1, 3, 9, 0.1]
		counts=[5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		##########################
		#counts=[0]*14

		#for i in range(len(self.corpus)):
		#	currcorpus=self.corpus[i]
		#	currcorpus.createLabels()
		#	gg=currcorpus.getlabels()
			
		#	for m in gg:
		#		for n in m:
		#			counts[n]+=1
		#	print(i)

		small=min(counts)

		weight=[]

		for k in counts:
			ww=(small/k)*1000
			weight.append(ww)

		self.weight=weight
		#################################


#################################################################
mm=proj(corpus)
mm.load_pretrain()
#mm.train()

testdata=[]

f = open("test.txt", "r", encoding="utf8")
lines = f.readlines()

paracount=0
flagg=False

for line in lines:
	if 'article_id:' in line:
		flagg=True

	if line=='\n':
		flagg=False

	if flagg==True and ('article_id:' not in line):
		newpara=paragraph(paracount)
		newpara.add_para(line.replace("\n",''))
		paracount+=1

	if '--------------------' in  line:
		testdata.append(newpara)

mm.evaluate(testdata)

#position id
#rule base
#same word
