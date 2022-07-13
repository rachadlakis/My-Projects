install.packages("arrow")

getwd()
setwd("E:/USJ/Projects/EDM_Big Data Formats Project")

#install the data from amazonaws.com
bucket <- "https://ursa-labs-taxi-data.s3.us-east-2.amazonaws.com"
for (year in 2009:2010) {
  if (year == 2019) {
    # We only have through June 2019 there
    months <- 1:6
  } else {
    months <- 1:12
  }
  for (month in sprintf("%02d", months)) {
    dir.create(file.path("nyc-taxi", year, month), recursive = TRUE)
    try(download.file(
      paste(bucket, year, month, "data.parquet", sep = "/"),
      file.path("nyc-taxi", year, month, "data.parquet"),
      mode = "wb"
    ), silent = TRUE)
  }
}

#open already installed libraires arrow and dplyr
library(arrow, warn.conflicts = FALSE)
library(dplyr, warn.conflicts = FALSE)


#open the parquet files with arrow library(by default the files opened will be parquet files also)
ds <- open_dataset("nyc-taxi", partitioning = c("year", "month"))

#The file format for open_dataset() is controlled by the format parameter,
# which has a default value of "parquet"


#see dataset details
ds

#Let' query the parquet files 
#The Files will not be fully imported to memory, see Environment

#Let's find the median tip percentage for rides with fares greater than $100 in 2009,
# broken down by the number of passengers:  12M record/month


# around 288M rows 19 columns each (approx. 11.5GB of data)
system.time(ds %>%
              filter(total_amount > 100, year==2009)%>%
              select(tip_amount, total_amount, passenger_count) %>%
              mutate(tip_pct = 100 * tip_amount / total_amount) %>%
              group_by(passenger_count) %>%
              collect() %>%
              summarise(
                median_tip_pct = median(tip_pct),
                n = n()
              ) %>%
              print() )


#second part csv file
#Let's query a csv dataset
 
df <-read.csv("10M Sales Records.csv")

#The time needed to query 12.5M records with 15 columns each (1.7 GB)

system.time(df %>% 
              filter(Unit.Price > 1, Country!= "LEBANON" ) %>%
              select(Region,  Total.Profit, Total.Cost)  %>% 
              mutate(Profit_Percenatge = Total.Profit/Total.Cost) %>%
              group_by(Region)  %>%
              collect() %>% 
              summarise(Total_Profit=sum(Total.Profit), n=n()) %>% 
              print())


#Nearly the same time for 10% of the data!!!
