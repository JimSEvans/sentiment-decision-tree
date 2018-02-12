library(rpart)
library(text2vec)
library(data.table)
library(rattle)
library(RColorBrewer)
library(rpart.plot)
library(arabicStemR)


# make sure your working directory is one that includes the data files
#   and no other text files. e.g. setwd('~/Source/sentiment-tree)
setwd('~/Source/sentimentDT')

# function to load the data (anything ending in ".txt") into a giant data frame
load_data <- function(path) { 
  files <- dir(path, pattern = '\\.txt', full.names = TRUE)
  tables <- lapply(files, 
                   read.csv, 
                   sep="\t", 
                   quote="", 
                   header=FALSE, 
                   colClasses = c("factor", "character"), 
                   col.names = c("label", "tweet_text"))
  do.call(rbind, tables)
}

#function to trim strings
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

all_data <- load_data(".")


#note: the same tweets sometimes were labeled independently by both 
# experts ("duplicate" tweets) to measure inter-annotater agreement
# so we must "de-duplicate"

set.seed(5) # set seed for reproducible results in random sampling

# shuffle rows, so that de-duplication (which will always throw out 
#  the second instance) doesn't favor keeping the label of one 
#  expert vs. another
all_data <- all_data[sample(nrow(all_data)),]

# get row indices of "duplicates"
duplicate_rows_indices <- duplicated(all_data$tweet_text) == TRUE
# remove "duplicates"
all_data <- all_data[duplicate_rows_indices == FALSE,]

# trim whitespace and capitalize labels
all_data$label <- toupper(trim(all_data$label))
all_data <- all_data[all_data$label %in% c("A","P","O","N"),]

#split training vs dev vs test data
lastTrainRowNum <- floor(nrow(all_data) * 0.7)
training_data <- all_data[1:lastTrainRowNum,]
lastDevRowNum <- floor(nrow(all_data) * 0.85)
dev_data <- all_data[lastTrainRowNum+1:lastDevRowNum,]
test_data <- all_data[lastDevRowNum+1:nrow(all_data),]

#add row ids
training_data$id<-seq.int(nrow(training_data))

# separate labels from data
data <- training_data[,c("id","tweet_text")]

# pre-process data, make vocab and vectorizer
it_train <- itoken(data$tweet_text, 
                   ids = data$id, 
                   progressbar = TRUE)

vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)

# make document-term matrix
dtm <- create_dtm(it_train, vectorizer)
# turn it into a data frame (rpart wants a data frame)
learning_data <- as.data.frame(as.matrix(dtm))

# origianal column names are "raw" terms as they appear in tweets.
#   That causes problems, so save them
columnNames <- colnames(learning_data)
# transliterate the arabic terms to latin alphabet, so that something can
#   be displayed in the final PDF of the tree (the library can't do arabic
#   at least without tinkering that I haven't tried)
columnNamesT <- transliterate(columnNames)

# name columns after integers 1 thru n
colnames(learning_data) <- seq.int(ncol(learning_data))

# add in the "label" column
learning_data$label <- as.factor(training_data$label)

# learn tree
tree <- rpart(label ~ . - label, # the "- label" part disallows the true label (y) to be used as a predictor (x) (crucial)
              data=learning_data,
              #max_depth = 5,
              #control = rpart.control(cp = 0.001),
              #control = rpart.control(minsplit=10),
              method="class")

# create a df for mapping training data column names used in the tree to
#  a transliterated (i.e. displayable) form for the PDF
words <- as.data.frame(matrix(columnNames))
words$V1 <- as.character(words$V1)
words$translit <- columnNamesT
colnames(words) <- c("orig","translit")
row_to_append <- data.frame(orig="<leaf>", translit="<leaf>")
row.names(row_to_append) <- "<leaf>"
words <- rbind(words, row_to_append)
new_var <- as.factor(words[as.character(tree$frame$var),][["translit"]])

rattle()

# make a tree object that is the same except arabic characters are transliterated (displayable)
alttree <- tree
alttree$frame$var <- new_var
fancyRpartPlot(alttree)

# 
# # ptree<- prune(tree, cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
# ptree<- prune(tree, cp=0.0016)
# fancyRpartPlot(ptree, uniform=TRUE, main="Decision tree for sentiment toward Da'esh")
# 
#numDims <- as.character(seq.int(ncol(learning_data)))
#colnames(learning_data) <- c(numDims, transliterate(colnames(learning_data)))

#learning_data[] <- lapply(learning_data, factor)
