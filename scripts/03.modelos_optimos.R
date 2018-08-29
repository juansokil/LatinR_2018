#limpio la memoria
rm( list=ls() )
gc()


####DEFINO PARAMETRO EMPRESA#####
## c:\program files\R\r-3.4.4\bin
##Rscript "C:\Users\Juan\Dropbox\fullmedia-classification\scripts\03.estimar_parametros.08.innovacion.R"  telefonica

getwd()
setwd('C:\\Users\\Usuarios CIO\\Documents\\fullmedia-classification\\scripts')
#setwd('C:/Users/Juan/Dropbox/fullmedia-classification/scripts')
#source('00.swd.R')
getwd()



#args <- commandArgs(TRUE)
#empresa <- args[1]
empresa='telefonica'
str(empresa)


library(readr)
library(xgboost)
library(AUC)

###################CALCULO VALORACION POSITIVA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_positiva_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)

parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree
#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_positiva_",empresa,".txt", sep ="")

campo_id             <-  "id"
clase_nomcampo       <-  "val_positiva"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_negativa","val_informativa", "tem_conducta", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 

##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#dataset <- dataset[ , !(names(dataset) %in%  "id")    ]

#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------

set.seed(102191)
modelo_val_positiva  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_positiva  = predict(modelo_val_positiva,  data.matrix(dataset) )
save(modelo_val_positiva, file = paste("../modelos/modelo_val_positiva_",empresa,".rda", sep=""))








###################CALCULO VALORACION NEGATIVA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_negativa_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree
#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_negativa_",empresa,".txt", sep ="")

campo_id             <-  "id"
clase_nomcampo       <-  "val_negativa"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_informativa", "tem_conducta", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")



#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$X.U.FEFF.Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_negativa  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_negativa  = predict(modelo_val_negativa,  data.matrix(dataset) )
save(modelo_val_negativa, file = paste("../modelos/modelo_val_negativa_",empresa,".rda", sep=""))





###################CALCULO VALORACION INFORMATIVA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_informativa_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree
#Parametros de entrada

#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_informativa_",empresa,".txt", sep ="")

campo_id             <-  "id"
clase_nomcampo       <-  "val_informativa"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$X.U.FEFF.Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_informativa  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_informativa  = predict(modelo_val_informativa,  data.matrix(dataset) )
save(modelo_val_informativa, file = paste("../modelos/modelo_val_informativa_",empresa,".rda", sep=""))







###################CALCULO VALORACION CONDUCTA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_conducta_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree

#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_conducta_",empresa,".txt", sep ="")


campo_id             <-  "id"
clase_nomcampo       <-  "tem_conducta"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_calidad", "val_informativa", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_conducta  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_conducta  = predict(modelo_val_conducta,  data.matrix(dataset) )
save(modelo_val_conducta, file = paste("../modelos/modelo_val_conducta_",empresa,".rda", sep=""))










###################CALCULO VALORACION CALIDAD##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_calidad_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree
#Parametros de entrada


#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_calidad_",empresa,".txt", sep ="")



campo_id             <-  "id"
clase_nomcampo       <-  "tem_calidad"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_calidad  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_calidad  = predict(modelo_val_calidad,  data.matrix(dataset) )
save(modelo_val_calidad, file = paste("../modelos/modelo_val_calidad_",empresa,".rda", sep=""))






















###################CALCULO VALORACION LIDERAZGO##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_liderazgo_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree


#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_liderazgo_",empresa,".txt", sep ="")




campo_id             <-  "id"
clase_nomcampo       <-  "tem_liderazgo"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_calidad", "tem_financiera", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
#variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_liderazgo  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_liderazgo  = predict(modelo_val_liderazgo,  data.matrix(dataset) )
save(modelo_val_liderazgo, file = paste("../modelos/modelo_val_liderazgo_",empresa,".rda", sep=""))















###################CALCULO VALORACION FINANCIERA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_financiera_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree

#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_financiera_",empresa,".txt", sep ="")



campo_id             <-  "id"
clase_nomcampo       <-  "tem_financiera"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_calidad", "tem_liderazgo", "tem_innovacion", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_financiera  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_financiera  = predict(modelo_val_financiera,  data.matrix(dataset) )
save(modelo_val_financiera, file = paste("../modelos/modelo_val_financiera_",empresa,".rda", sep=""))





###################CALCULO VALORACION INNOVACION##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_innovacion_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree


#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_innovacion_",empresa,".txt", sep ="")


campo_id             <-  "id"
clase_nomcampo       <-  "tem_innovacion"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_interno", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_innovacion  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_innovacion  = predict(modelo_val_innovacion,  data.matrix(dataset) )
save(modelo_val_innovacion, file = paste("../modelos/modelo_val_innovacion_",empresa,".rda", sep=""))








###################CALCULO VALORACION INTERNO##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_interno_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree

#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_interno_",empresa,".txt", sep ="")



campo_id             <-  "id"
clase_nomcampo       <-  "tem_interno"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_sustenta")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_interno  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_interno  = predict(modelo_val_interno,  data.matrix(dataset) )
save(modelo_val_interno, file = paste("../modelos/modelo_val_interno_",empresa,".rda", sep=""))



###################CALCULO VALORACION SUSTENTA##################
parametros <- read_delim(paste("../parametros/xgboost_MBO_AUC_salida_val_sustenta_",empresa,".txt",sep=""), "\t", escape_double = FALSE, trim_ws = TRUE)
parametros$diferencia =  (parametros$AUC + parametros$AUC_train)

#selecciono el modelo con menos diferencia entre train y test
diferencia_min <- max(parametros$diferencia)
optimos <- subset(parametros, diferencia == diferencia_min)
#Aqui van los parametros optimos, los mejores de TODO  Grid Search y Bayesian Search
#( los valores concretos de este codigo estan de ejemplo, y NO son necesariamente los optimos )
pnround           <- optimos$nround
pmin_child_weight <-   optimos$min_child_weight
pmax_depth        <-   optimos$max_depth
peta              <-   optimos$eta
palpha            <-   optimos$alpha
plambda           <-   optimos$lambda
pgamma            <-   optimos$gamma
pcolsample_bytree <-   optimos$colsample_bytree

#Parametros de entrada
archivo_entrada <- paste("../Bases/","base.train_",empresa,".txt", sep ="")
archivo_importantes <- paste("../bases/","xgboost_importancia_sustenta_",empresa,".txt", sep ="")



campo_id             <-  "id"
clase_nomcampo       <-  "tem_sustenta"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva","val_negativa", "tem_conducta", "val_informativa", "tem_calidad", "tem_liderazgo", "tem_financiera", "tem_innovacion", "tem_interno")


#leo el dataset
dataset <- read.table( archivo_entrada, header=TRUE, sep="\t", row.names=campo_id, encoding="latin1")

#dejo la clase en {0,1}
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <-   dataset[ , !(names(dataset) %in% c(clase_nomcampo) ) ] 
##Dejo solo los casos importantes
variables_importantes <- read.table( archivo_importantes, header=TRUE, encoding="UTF-8")
#dataset_sinclase <- dataset_sinclase[ , (names(dataset_sinclase) %in%   variables_importantes$Feature  )    ] 
#genero formato entrada para xgboost
dtrain = xgb.DMatrix(data = data.matrix(dataset_sinclase),   label = dataset[, clase_nomcampo], missing = NA )

#------------------------------------------------------


set.seed(102191)
modelo_val_sustenta  = xgboost(data = dtrain, missing = NA , subsample = 1.0,  eta = peta,  colsample_bytree = pcolsample_bytree,  min_child_weight = pmin_child_weight,  max_depth = pmax_depth, alpha = palpha, lambda = plambda, gamma = pgamma, objective="binary:logistic", eval_metric = "auc", maximize =TRUE, nround= pnround, base_score = kbase_score)
probabilidades_val_sustenta  = predict(modelo_val_sustenta,  data.matrix(dataset) )
save(modelo_val_sustenta, file = paste("../modelos/modelo_val_sustenta_",empresa,".rda", sep=""))


