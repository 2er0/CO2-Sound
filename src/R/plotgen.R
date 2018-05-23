options(stringsAsFactors = F)

data = read.csv("~/uni/co2/CO2-Sound/data/resultTable.csv")

library(ggplot2)
library(reshape2)

cnns = data[data$model=="cnn", ]
# cnns$model_version = paste0(cnns$model, "_", cnns$version)

cnns.melted = melt(cnns, id.vars = c("model", "source", "version"))
cnns.melted$source_version = paste0(cnns.melted$source, "_", cnns.melted$version)
cnns.melted = cnns.melted[!is.element(cnns.melted$variable, c("model_version")),]

ggplot(cnns.melted, aes(variable, value)) + 
  geom_col(aes(fill=source_version), position = position_dodge())

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


ggplot(data, aes(AccTrain, LossTrain)) +
  geom_smooth(aes(color= model), method = "lm", formula = y~x) + 
  geom_point(aes(color = model), size = 3) +

  # scale_color_manual(values=cbPalette ) + 
  ggtitle("Training: Accuracy vs. Loss")


ggplot(data, aes(AccVal, LossVal)) + 
  geom_point(aes(color = model), size = 3) + 
  ggtitle("Validation: Accuracy vs. Loss") 

cor (data$AccVal, data$LossVal)
ggplot(data, aes(AccTest, LossTest)) +
  geom_smooth(aes(color= model), method = "lm", formula = y~x) + 
  geom_point(aes(color = model), size = 3) + ggtitle("Test: Accuracy vs. Loss")





data.3 = data[1:6, ]
data.3$distiction = paste0(data.3$model, "_", data.3$source, "_",data.3$version )
data.3.tomelt = data.3[,c("distiction", "AccTrain", "AccVal", "AccTest")]

melted.data.3 = melt(data.3.tomelt, id.vars = "distiction")
ggplot(melted.data.3, aes(distiction,  value)) + 
  geom_line(aes(color= variable, group=variable))
