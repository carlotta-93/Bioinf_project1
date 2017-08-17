

subset_data_best_indiv <-subset(data_from_exp_best_indiv, select=time:molarity)
subset_data_best_indiv

write.csv2(subset_data_best_indiv, file='best_indiv_found.csv')