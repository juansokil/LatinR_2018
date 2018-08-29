#XGBoost
#cross validation  
#Optimiza  AUC

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
library(xgboost)
library(Matrix)
library(mlrMBO)



empresa='nombre_empresa'
str(empresa)



programa             <-  "estimar_parametros.cat.positiva.r"
algoritmo            <-  "XGBoost"
busqueda             <-  "MBO"
estimacion           <-  "crossvalidation"
archivo_entrada <- paste("../bases/","base.train_",empresa,".txt", sep ="")
campos_separador     <-  "\t"
campo_id             <-  "id"
clase_nomcampo       <-  "val_positiva"
clase_valor_positivo <-  "1"
campos_a_borrar      <-  c("val_positiva")


archivo_salida <- paste("../","parametros/xgboost_MBO_AUC_salida_val_positiva_",empresa,".txt", sep ="")
#archivo_importantes <- paste("../","bases/xgboost_importancia_positiva_",empresa,".txt", sep ="")

archivo_trabajo      <-  "./cloud1/Text_Classification_01.RDATA"
#archivo_trabajo      <-  "./Text_Classification/Text_Classification.RDATA"
archivo_grid         <-  "./xgboost_MBO_AUC_grid_positivo.txt"
archivo_grid_faltan  <-  "./xgboost_MBO_AUC_grid_positivo_faltan.txt"
puntos_iniciales     <-    50
iteraciones          <-  1000
crossvalidation_folds <- 5





#cargo los datos
dataset <- read.table( archivo_entrada, header=TRUE, sep=campos_separador, row.names=campo_id, encoding="latin1")
#borro las variables que no me interesan
dataset <- dataset[ , !(names(dataset) %in%   campos_a_borrar  )    ] 


#dejo la clase en {0,1} 
dataset[,clase_nomcampo] <-  as.numeric( dataset[,clase_nomcampo] == clase_valor_positivo  )
kbase_score  <-  sum( dataset[,clase_nomcampo] ) / length(dataset[,clase_nomcampo])
dataset_sinclase <- dataset[ , !(names(dataset) %in%   c( clase_nomcampo)  )    ]


####Armo la base para Xgboost#####
dtrain   <- xgb.DMatrix( data = data.matrix(dataset_sinclase),  label = dataset[ , clase_nomcampo], missing=NA )




#####---------------------------------FUNCIONES---------------------

evolucion_refresh  = function()
{
	evolucion <- data.frame( "tiempo"=double(),  "auc_max"=double() )

	salida <-  read.table( archivo_salida, header=TRUE, sep=campos_separador, encoding="latin1" )
	salida.largo  <-  nrow( salida )

	if( salida.largo >= 1 )
	{
		evolucion[ 1, "tiempo" ]        <-  salida[ 1, "tiempo_promedio" ] /(3600*24)
		evolucion[ 1, "auc_max" ]  <-  salida[ 1, "AUC" ]

		
		if(  salida.largo > 1 )
		{
		for( i in 2:salida.largo )
		{
		    evolucion[ i, "tiempo" ]       <-   evolucion[ i-1, "tiempo" ] + salida[ i, "tiempo_promedio" ]/(3600*24)
		    evolucion[ i, "auc_max" ] <-   max(  evolucion[ i-1, "auc_max" ],  salida[ i, "auc" ]  )
		}
		}



	}

}
#------------------------------------------------------


generar_grid_inicial =  function( ppuntos_iniciales, pfun )
{

	if( !file.exists( archivo_grid) )
	{
	   cat( "pcolsample_bytree",   "\t",
                "peta",                "\t", 
		"palpha",              "\t",
		"plambda",             "\t", 
		"pgamma",              "\t", 
		"pmin_child_weight",   "\t", 
		"pmax_depth",          "\t", 
		"y",                   "\n", 
		file=archivo_grid, fill=FALSE, append=FALSE 
    	      )	 
	   design <- generateDesign( puntos_iniciales, getParamSet(pfun), fun = lhs::maximinLHS)
	   write.table( design, archivo_grid_faltan, sep=campos_separador, row.names=FALSE )
	}

	design <-  read.table( archivo_grid_faltan, header=TRUE, sep=campos_separador, encoding="latin1" )
	design.largo = nrow( design)

	if( design.largo > 0 )
	{
	for(  i in 1:design.largo )
	{
	   linea <- as.list( design[ 1, ]  )
	   xgboost_auc(  linea  )
	   design  <-  design[ -1 , ]
	   write.table( design, archivo_grid_faltan, sep=campos_separador, row.names=FALSE )
	}
	}

}
#------------------------------------------------------

xgboost_auc <- function( x = list( pcolsample_bytree,   peta,  palpha, plambda, pgamma,
                          		   pmin_child_weight, pmax_depth) )
{
	# Aqui va un nro grande de rounds, esperando que corte antes
	vnround            <-  20000
	# Este parametro esta fijado 
	vsubsample   <-    1.0

	set.seed( 102191 )
	t0       <-  Sys.time()
        modelo.cv = xgb.cv( 
				data = dtrain,  
				missing = NA,
				stratified = TRUE,       nfold = crossvalidation_folds ,  # cross-validation
				objective="binary:logistic",
				nround= vnround, early_stopping_rounds = 100,
				base_score = kbase_score ,
				metric = "auc",            maximize =TRUE,
				subsample = vsubsample, 
	 			colsample_bytree = x$pcolsample_bytree, 
		                eta = x$peta,
 				min_child_weight = x$pmin_child_weight, 
	 			max_depth = x$pmax_depth,
		 		alpha = x$palpha, lambda = x$plambda, gamma = x$pgamma
			)

	t1       <-  Sys.time()
	tiempo   <- as.numeric(  t1 - t0, units = "secs")
	
	auc_max       <- max( modelo.cv$evaluation_log[ , test_auc_mean] )
	iteracion_max <- which.max(  modelo.cv$evaluation_log[ , test_auc_mean] )
	
	auc_exportar  <- - auc_max
	auc_max_train       <- max( modelo.cv$evaluation_log[ , train_auc_mean] )
	auc_exportar_train  <- - auc_max_train
	
	 cat( 	x$pcolsample_bytree,  "\t",  x$peta,               "\t", x$palpha,             "\t",  x$plambda,            "\t", x$pgamma,             "\t", x$pmin_child_weight,  "\t", x$pmax_depth,         "\t", auc_exportar,         "\n",  file=archivo_grid, fill=FALSE, append=TRUE) 

	cat(auc_max, auc_exportar_train, tiempo, iteracion_max,  vsubsample, x$pcolsample_bytree, x$peta, x$palpha, x$plambda, x$pgamma, x$pmin_child_weight, x$pmax_depth,
	      format(Sys.time(), "%Y%m%d %H%M%S"), archivo_entrada, clase_nomcampo, programa, algoritmo, busqueda, estimacion,
	      "\n", sep="\t", file=archivo_salida, fill=FALSE, append=TRUE )
	evolucion_refresh( )
   return ( auc_exportar )
}
#------------------------------------------------------
#escribo los  titulos  del archivo salida
if( !file.exists( archivo_salida) )
{
 cat("AUC", "AUC_train","tiempo_promedio","nround","subsample","colsample_bytree","eta","alpha","lambda","gamma","min_child_weight","max_depth",	"fecha", "dataset", "clase", "programa", "algoritmo", "busqueda" , "estimacion", 
	"\n", sep="\t", file=archivo_salida, fill=FALSE, append=FALSE )
}


configureMlr(show.learner.output = FALSE)
obj.fun = makeSingleObjectiveFunction(
		name = "prueba",
		fn   = xgboost_auc,
		par.set = makeParamSet(
			makeNumericParam("pcolsample_bytree" ,  lower=0.05    , upper= 1.0),
			makeNumericParam("peta"              ,  lower=0.0     , upper= 0.1),
			makeNumericParam("palpha"            ,  lower=0.0     , upper= 1.0),
			makeNumericParam("plambda"           ,  lower=0.0     , upper= 1.0),
			makeNumericParam("pgamma"            ,  lower=0.0     , upper=20.0),
			makeNumericParam("pmin_child_weight" ,  lower=0.0     , upper=10.0),
			makeIntegerParam("pmax_depth"        ,  lower=2L      , upper=5L)
		),
		has.simple.signature = FALSE,
		global.opt.value = -1
		)

ctrl = makeMBOControl(propose.points = 1L )
ctrl = setMBOControlTermination(ctrl, iters = iteraciones )
ctrl = setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI(),
			opt  = "focussearch", opt.focussearch.points = iteraciones )

#nugget.stability - es un problema de la libreria DiceKriging#####
surr.km = makeLearner("regr.km", predict.type = "se", covtype = "matern3_2", nugget.stability=1e-08, control = list(trace = FALSE))
lrn  = makeMBOLearner(ctrl, obj.fun)


evolucion_refresh()
#Generar Grid Inicial - Solo corre la primera vez, cuando crea el archivo
generar_grid_inicial(puntos_iniciales, obj.fun )
design <-  read.table( archivo_grid, header=TRUE, sep=campos_separador, encoding="latin1" )
res = mbo(obj.fun, design = design, learner = surr.km, control = ctrl)








