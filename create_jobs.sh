# create job.yaml files
for i in $(eval echo {1..$1})
do
  cat iris-job-template.yml | sed "s/\$ITEM/$i/" > ./hyperparam-jobs-specs/iris-job-$i.yml
done
