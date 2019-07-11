if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, ggplot2,corrplot,plotly,ggfortify, textclean,doParallel,polycor,tensorflow,gam,lubridate,
               GGally, readr,caret,tidyr,reshape,rstudioapi,kknn,randomForest,DescTools)

current_path <- getActiveDocumentContext()
setwd(dirname(dirname(current_path$path)))
rm(current_path)

cl <- detectCores()
cl <- makeCluster(cl/2)
registerDoParallel(cl)

trainset <- read_csv("Datasets/dengue_features_train.csv")
testset <- read_csv("Datasets/dengue_features_test.csv")
labels <- read_csv("Datasets/dengue_labels_train.csv")
trainset$total_cases <- labels$total_cases
trainset$ID <- c(1:nrow(trainset))
testset$ID <- c(1:nrow(testset))
labels$ID <- c(1:nrow(labels))
trainset$month <- as.factor(month(trainset$week_start_date))
testset$month <- as.factor(month(testset$week_start_date))
# trainset$quarter <- as.factor(quarter(trainset$week_start_date,fiscal_start = 3))
# testset$quarter <- as.factor(quarter(testset$week_start_date,fiscal_start = 3))

changevars <- function(x){
trainset <- x
trainset$avgtemp <- (trainset$reanalysis_air_temp_k + trainset$reanalysis_avg_temp_k)/2 - 273.15
trainset$avgtemp <- (trainset$station_avg_temp_c + trainset$avgtemp)/2
trainset$dewpoint <- trainset$reanalysis_dew_point_temp_k - 273.15
# trainset$maxtemp <- (trainset$station_max_temp_c + trainset$reanalysis_max_air_temp_k - 273.15)/2
# trainset$mintemp <- (trainset$station_min_temp_c + trainset$reanalysis_min_air_temp_k - 273.15)/2

trainset$station_diur_temp_rng_c <- NULL
trainset$reanalysis_tdtr_k <- NULL
trainset$reanalysis_air_temp_k <- NULL
trainset$reanalysis_avg_temp_k <- NULL
trainset$station_avg_temp_c <- NULL
trainset$reanalysis_dew_point_temp_k <- NULL
trainset$station_max_temp_c <- NULL
trainset$reanalysis_max_air_temp_k <- NULL
trainset$station_min_temp_c <- NULL
trainset$reanalysis_min_air_temp_k <- NULL
trainset$ndvi_ne <- NULL
trainset$ndvi_nw <- NULL
trainset$ndvi_se <- NULL
trainset$ndvi_sw <- NULL
trainset$reanalysis_sat_precip_amt_mm <- NULL
return(trainset)
}
scalekelvin <- function(x){
  df <- x
  df$reanalysis_avg_temp_k <-  df$reanalysis_avg_temp_k - 273.15
  df$reanalysis_max_air_temp_k <- df$reanalysis_max_air_temp_k - 273.15
  df$reanalysis_tdtr_k <- df$reanalysis_tdtr_k - 273.15
  df$reanalysis_air_temp_k <-  df$reanalysis_air_temp_k - 273.15
  df$reanalysis_dew_point_temp_k <-  df$reanalysis_dew_point_temp_k - 273.15
  df$reanalysis_min_air_temp_k <-  df$reanalysis_min_air_temp_k - 273.15
  df$station_precip_mm <-  NULL
  return(df)
}
bycity <- split(trainset,trainset$city)
#####Sanjuantrain#####
sanjuan <- bycity$sj
sanjuanmonth <- split(sanjuan,sanjuan$month)
sanjuanmonth <- lapply(sanjuanmonth, na.gam.replace)
# sanjuanmonth <- lapply(sanjuanmonth, changevars)
sanjuan <- bind_rows(sanjuanmonth)
sanjuan <- scalekelvin(sanjuan)
sanjuanpreprocess <- preProcess(x = sanjuan[,c(5:23,26)],thresh = 0.9,method = c("center","scale","pca"))
sanjuanpca <- predict(sanjuanpreprocess,sanjuan)
#####Iquitostrain#####
iquitos <- bycity$iq
iquitosmonth <- split(iquitos,iquitos$month)
iquitosmonth <- lapply(iquitosmonth, na.gam.replace)
# iquitosmonth <- lapply(iquitosmonth, changevars)
iquitos <- bind_rows(iquitosmonth)
iquitos <- scalekelvin(iquitos)
iquitospreprocess <- preProcess(x = iquitos[,c(5:23,26)],thresh = 0.9,method = c("center","scale","pca"))
iquitospca <- predict(iquitospreprocess,iquitos)

bycitytest <- split(testset,testset$city)
#####Sanjuantest#####
sanjuantest <- bycitytest$sj
sanjuantest$total_cases <- sanjuantest$ID
sanjuantestmonth <- split(sanjuantest,sanjuantest$month)
sanjuantestmonth <- lapply(sanjuantestmonth, na.gam.replace)
# sanjuantestmonth <- lapply(sanjuantestmonth, changevars)
sanjuantest <- bind_rows(sanjuantestmonth)
sanjuantest <- scalekelvin(sanjuantest)
sanjuantestpca <- predict(sanjuanpreprocess,sanjuantest)
#####Iquitostest#####
iquitostest <- bycitytest$iq
iquitostest$total_cases <- iquitostest$ID
iquitostestmonth <- split(iquitostest,iquitostest$month)
iquitostestmonth <- lapply(iquitostestmonth, na.gam.replace)
# iquitostestmonth <- lapply(iquitostestmonth, changevars)
iquitostest <- bind_rows(iquitostestmonth)
iquitostest <- scalekelvin(iquitostest)
iquitostestpca <- predict(iquitospreprocess,iquitostest)

treecontrol <- trainControl(method = "cv",number = 5)
sanjuanlm <- train(sanjuanpca[,c(7:14)],
                              y = sanjuanpca$total_cases,method = "rf",trControl = treecontrol)
iquitoslm <- train(iquitospca[,c(7:14)],
                              y = iquitospca$total_cases,method = "rf",trControl = treecontrol)

# for (i in names(sanjuantestmonth)) {
#   sanjuantestmonth[[i]]$total_cases <- round(predict(sanjuanlm[[i]],sanjuantestmonth[[i]]))
#   iquitostestmonth[[i]]$total_cases <- round(predict(iquitoslm[[i]],iquitostestmonth[[i]]))
# }

sanjuantestpca$total_cases <- round(predict(sanjuanlm,sanjuantestpca))
iquitostestpca$total_cases <- round(predict(iquitoslm,iquitostestpca))
results <- rbind(sanjuantestpca,iquitostestpca)
results <- arrange(results,results$ID)
submission <- as.data.frame(cbind(city = results$city,year = results$year,weekofyear = results$weekofyear,total_cases = results$total_cases))
write_csv(submission,path = "Datasets/Submission3Joan.csv")
 