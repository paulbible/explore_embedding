#!/bin/env Rscript

# set correct directory
setwd('C:\\Users\\Paul\\PycharmProjects\\explore_embedding\\src')

# Load control and approval variables
r.vars <- read.csv('../data/regression_control_vars.csv',row.names=1)
# variables for concurrent approval
vars.curr <- r.vars[,-c(10,9,8)]
approval.curr <- r.vars[,9]
# variables for 6 months out approval
vars.6m <- r.vars[,-c(10,9,7)]
approval.6m <- r.vars[,10]

# load cluster matrix
c.mat <- read.csv('../data/regression_count_matrix.csv',row.names=1)

# all data
full.mat.curr <- as.matrix(cbind(c.mat, vars.curr))
mod <- lm(approval.curr ~ full.mat.curr)
summary(mod)

full.mat.6m <- as.matrix(cbind(c.mat, vars.6m))
mod <- lm(approval.6m ~ full.mat.6m)
summary(mod)
