#!/bin/bash

#SBATCH --job-name=KDD12-Test
#SBATCH --nodes=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --mail-user=ncm5@rice.edu
#SBATCH --mail-type=all
#SBATCH -time=00:30:00
export PROJECTS=/projects/as143/ncm5/benchmark
export HADOOP_HOME=$PROJECTS/hadoop-2.8.5
export PATH=$HADOOP_HOME/bin:$PROJECTS/myhadoop/bin:$PATH:$PATH
export JAVA_HOME=/opt/apps/software/Core/Java/12.0.2
export HADOOP_CONF_DIR=$PWD/hadoop-conf.$SLURM_JOBID
export SPARK_BIN=$PROJECTS/spark-2.2.0-bin-hadoop2.6/bin
export HDFS_HOME=/user/ncm5
myhadoop-configure.sh -s /tmp/$USER/$SLURM_JOBID
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
$HADOOP_HOME/bin/hdfs dfs -mkdir -p $HDFS_HOME/data
$HADOOP_HOME/bin/hdfs dfs -ls $HDFS_HOME/data
$HADOOP_HOME/bin/hdfs dfs -put kdd12-data $HDFS_HOME/data
$HADOOP_HOME/bin/hdfs dfs -put kdd12-query $HDFS_HOME/data
$SPARK_BIN/spark-submit --master yarn --deploy-mode cluster --driver-memory 4g --executor-memory 2g --executor-cores 1 lsh.py
$HADOOP_HOME/bin/hdfs dfs -get $HDFS_HOME/data/output_directory
$HADOOP_HOME/sbin/stop-dfs.sh
$HADOOP_HOME/sbin/stop-yarn.sh
myhadoop-cleanup.sh