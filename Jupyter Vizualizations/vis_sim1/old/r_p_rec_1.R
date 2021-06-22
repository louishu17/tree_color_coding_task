input <- read.csv(file = "fin.csv")
df <- data.frame(input)
df_new <- data.frame(df$p, df$rec)
colnames(df_new) <- c("p", "rec")
print(df_new)
attach(df_new)
plot(p, rec, main="Increasing p Increases Return Value",
     xlab="p", ylab="rec", pch=19)
detach(df_new)