library(ggplot2)

file <- read.csv("/Users/jameskitchens/Documents/GitHub/sparg2.0/jupyter_notebooks/benchmarking/benchmarking_with_td.csv", colClasses=c(algo_order="character"))
file$paths_sum <- round(file$paths_sum)
file$paths_modified_sum <- round(file$paths_modified_sum)
file$hybrid_nr_sum <- round(file$hybrid_nr_sum)


filtered <- file[which(file$paths_sum!=file$hybrid_nr_sum),]

filtered$paths_sum
filtered$paths_modified_sum

ggplot(data=file) +
  geom_point(aes(paths, paths_time, color="Paths"), alpha=0.3) +
  geom_smooth(aes(paths, paths_time, color="Paths"), se=F) +
  geom_point(aes(paths, hybrid_r_time, color="Hybrid (Recursive)"), alpha=0.3) +
  geom_smooth(aes(paths, hybrid_r_time, color="Hybrid (Recursive)"), se=F) +
  geom_point(aes(paths, paths_modified_time, color="Paths (Modified)"), alpha=0.3) +
  geom_smooth(aes(paths, paths_modified_time, color="Paths (Modified)"), se=F) +
  xlab("Number of Paths in ARG") +
  ylab("Time to Compute Covariance Matrix") +
  theme_minimal()

#geom_label(aes(paths, 250, label=paste("#Samples:", num_samples, "\n#Trees:", num_trees))) +


ggplot(data=file) +
  geom_point(aes(num_trees, paths_time), color="red") +
  geom_point(aes(num_trees, hybrid_r_time), color="blue") +
  geom_point(aes(num_trees, paths_modified_time), color="black") +
  theme_minimal()


ggplot(data=file) +
  geom_point(aes(num_samples, paths_time), color="red") +
  geom_point(aes(num_samples, hybrid_r_time), color="blue") +
  geom_point(aes(num_samples, hybrid_nr_time), color="black") +
  geom_label(aes(num_samples, 500, label=algo_order)) +
  geom_label(aes(num_samples, 400, label=paths)) +
  ylim(0, 500) +
  theme_minimal()


library(tidyverse)

topology <- read.csv("/Users/jameskitchens/Documents/GitHub/sparg2.0/jupyter_notebooks/benchmarking/topology.csv")

top_agg <- topology %>%
  group_by(num_samples, sequence_length) %>%
  summarize(
    num_nodes=median(nodes),
    num_paths=median(num_paths),
    num_loop_groups=median(num_loop_groups),
    max_loop_size=median(max_loop_size)
  ) %>%
  pivot_longer(
    cols = !c(num_samples, sequence_length),
  )

ggplot(data=top_agg) +
  geom_point(aes(num_samples, value, color=sequence_length)) +
  theme_minimal() +
  facet_wrap(~name, scale="free")



benchmark <- read.csv("/Users/jameskitchens/Downloads/benchmarking_new.csv")

ggplot(data=benchmark) +
  geom_smooth(aes(paths, paths_time), color="red") +
  geom_smooth(aes(paths, hybrid_r_time), color="blue") +
  geom_smooth(aes(paths, hybrid_nr_time), color="black") +
  #geom_label(aes(paths, 250, label=paste("#Samples:", num_samples, "\n#Trees:", num_trees))) +
  xlab("Number of Paths in ARG") +
  ylab("Time to Compute Covariance Matrix") +
  theme_minimal()
