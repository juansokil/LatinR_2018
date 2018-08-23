
#limpio la memoria
rm( list=ls() )
gc()

#############LIBRERIAS#############
library(text2vec)
library(data.table)
library(magrittr)
library(readr)
library(dplyr)
library(stringr)
library(NLP)
library(tm)
library(quanteda)
library(caret)
library(stringr)
library(qdapRegex)
library(DBI)
library(RMySQL)
library(Hmisc)


getwd()

empresa='nombre_empresa'
str(empresa)




###LOAD DATABASE#############





##SEPARA TRAIN - TEST
base.train<-base[1:round(nrow(base)*0.80),]
base.test<-base[(round(nrow(base)*0.80)+1):nrow(base),]
base <- base.train
test <- base.test




#Concatenar los 2 textos, titulo y nota
base$texto <- paste(base$note_title, base$note_text)

#Genera corpus
vs <- VectorSource(base$texto)
corp <- Corpus(vs)
corp <- tm_map(corp, tolower)
corp <- tm_map(corp, removeNumbers)
corp <- tm_map(corp, stripWhitespace)
corp <- tm_map(corp, removePunctuation)
corp <- tm_map(corp, removeWords, stopwords("spanish"))
corp <- tm_map(corp, stemDocument, language = "spanish")


###Arma la base
base$textobien <- sapply(corp, paste, collapse = " ")

#Cambia el encoding#######
base$textobien <- iconv(base$textobien, "UTF-8", "latin1")

############Tokenizar el texto#################
prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(base$textobien, preprocessor = prep_fun, tokenizer = tok_fun, ids = base$id, progressbar = TRUE)

#Unigramas - Bigramas
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))

###############Poda del vocabulario######### según frecuencia de aparición de palabras#####
pruned_vocab <- prune_vocabulary(vocab,doc_proportion_max = 0.8, doc_proportion_min = 0.03)


###############Poda del vocabulario######### según largo de palabras#####
pruned_vocab$largo <- nchar(pruned_vocab$term, type = "chars", allowNA = FALSE, keepNA = NA)
pruned_vocab <- pruned_vocab[ which(pruned_vocab$largo > 3), ]

####Construye Matriz TF####
vectorizer <-  vocab_vectorizer(pruned_vocab)
dtm_train  <- create_dtm(it_train, vectorizer)

####Construye Matriz TF-IDF####
tfidf = TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)



##PREPARA PARA XGBOOST
base.train <- as.data.frame(data.matrix(dtm_train_tfidf))

##COSTRUYE VARIABLES DICOTOMICAS
base.train$val_positiva <- 0
base.train$val_negativa <- 0
base.train$val_informativa <- 0
base.train$val_positiva[base$valoracion =='Positivo'] <- 1
base.train$val_negativa[base$valoracion =='Negativo'] <- 1
base.train$val_informativa[base$valoracion =='Informativo'] <- 1

#Aleatorizar los datos
#base.train <- base.train[sample(1:nrow(base.train)), ]
#base.train <- base.train[,sample(ncol(base.train))]


## DEFINE EL DIRECTORIO DE TRABAJO Y GUARDA LOS DATOS
setwd("../Modelos")

pruned <- paste("pruned_vocab_",empresa,".rda",sep="")
vectorized <- paste("vectorizer_",empresa,".rda",sep="")
tfidfed <- paste("tfidf_",empresa,".rda",sep="")

save(pruned_vocab, file =pruned)
save(vectorizer, file =vectorized)
save(tfidf, file =tfidfed)

setwd("../Bases")

base_train_bajar <- paste("base.train_",empresa,".txt", sep ="")
base_empresa <- paste("base_",empresa,".txt", sep ="")

write.table(base.train,base_train_bajar,sep="\t",row.names=FALSE)
write.table(base,base_empresa,sep="\t",row.names=FALSE)




