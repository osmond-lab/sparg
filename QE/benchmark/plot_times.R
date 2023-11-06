library(ggplot2)

file <- read.csv("benchmarking_final.csv")

ggplot(data=file) +
  geom_point(aes(paths, paths_time, color="Paths"), alpha=0.5) +
  geom_smooth(aes(paths, paths_time, color="Paths"), se=F) +
  geom_point(aes(paths, bottom_up_time, color="Bottom Up"), alpha=0.5) +
  geom_smooth(aes(paths, bottom_up_time, color="Bottom Up"), se=F) +
  scale_color_manual(values=c("#999999", "#E69F00")) +
  #geom_point(aes(paths, minimal_time, color="Minimal"), alpha=0.3) +
  #geom_smooth(aes(paths, minimal_time, color="Minimal"), se=F) +
  #coord_cartesian(ylim=c(0,0.5)) + 
  #xlim(min=0, max=1000) +
  #ylim(min=0, max=20) +
  xlab("Number of Paths in ARG") +
  ylab("Time to Compute Covariance Matrix (seconds)") +
  labs(color="Method") +
  theme_minimal()


image <- ggplot(data=file) +
  geom_point(aes(paths, paths_time, color="Naive"), alpha=0.5) +
  geom_smooth(aes(paths, paths_time, color="Naive"), se=F) +
  geom_point(aes(paths, bottom_up_time, color="Optimized"), alpha=0.5) +
  geom_smooth(aes(paths, bottom_up_time, color="Optimized"), se=F) +
  geom_point(aes(paths, minimal_time, color="Minimal (new)"), alpha=0.3) +
  geom_smooth(aes(paths, minimal_time, color="Minimal (new)"), se=F) +
  scale_color_manual(breaks=c("Naive", "Optimized", "Minimal (new)"), values=c("#E69F00", "#56B4E9", "#999999")) +
  coord_cartesian(ylim=c(0,0.1)) + 
  #xlim(min=0, max=1000) +
  ylim(min=0, max=5) +
  xlab("Number of Paths in ARG") +
  ylab("Time to Compute Covariance Matrix (seconds)") +
  #labs(color="Methods") +
  theme_minimal() +
  theme(legend.position = "none")

image

ggsave(file="benchmark.svg", plot=image, width=5, height=5)
