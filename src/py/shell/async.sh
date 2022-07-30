#!/bin/bash
server=
client_list=(

)

# server args
fraction_fit=${1:-0.1}
fraction_eval=${2:-0.1}
min_fit_clients=${3:-2}
min_eval_clients=${4:-2}
rounds=${5:-3}
min_available_clients=${6:-2}
alpha=${7:-0.5}
staleness=${10:-1}
strategy=${11:-0}
# client args
log_n=${8:-0}
e=${9:-10}

# check args
echo $fraction_fit $fraction_eval $min_fit_clients $min_eval_clients $rounds $min_available_clients $log_n


# server activation
echo pick server ${server}
startTime=$(date "+%Y-%m-%d %H:%M:%S")
start_seconds=$(date --date="$startTime" +%s);
nohup ssh $server "
	export PYTHONPATH=/opt/flower/src/py;
	cd /opt/flower/src/py/flwr_experiment/gcn_cora;
	python3 async.py --strategy ${strategy} --staleness ${staleness} --alpha ${alpha} --fraction_fit ${fraction_fit} --fraction_eval ${fraction_eval} --min_fit_clients ${min_fit_clients} --min_eval_clients ${min_eval_clients} --min_available_clients ${min_available_clients} --rounds ${rounds} ; echo server is ready;">logs/server.log &

echo waiting for server to activate
sleep 5

# client activation
for(( i=0;i<${min_available_clients};i++)) do
	echo ${client_list[i]} activating in nohup
	nohup ssh ${client_list[i]} "echo ${client_list[i]} is activating ;
	export PYTHONPATH=/opt/flower/src/py;
	cd /opt/flower/src/py/flwr_experiment/gcn_cora;
	python3 client${i}.py --server_address ${server}:8080 --n ${log_n} --e ${e};">logs/client${i}.log &

done
wait

# readout all information to a experiment result file for further processcd
currentTime=$(date "+%Y-%m-%d-%H:%M:%S")
endTime=$(date "+%Y-%m-%d %H:%M:%S")
end_seconds=$(date --date="$endTime" +%s);
duration=$((end_seconds-start_seconds))

echo currentTime : ${currentTime}
echo execute time ${duration}s >> result/result-${currentTime}.log
cd result
touch result-${currentTime}.log
cd ..
# args input write in
echo fraction_fit ${fraction_fit} >> result/result-${currentTime}.log
echo fraction_eval ${fraction_eval} >> result/result-${currentTime}.log
echo min_fit_clients ${min_fit_clients} >> result/result-${currentTime}.log
echo min_eval_clients ${min_eval_clients} >> result/result-${currentTime}.log
echo rounds ${rounds} >> result/result-${currentTime}.log
echo min_available_clients ${min_available_clients} >> result/result-${currentTime}.log
echo alpha ${alpha} >> result/result-${currentTime}.log
echo log_n ${log_n} >> result/result-${currentTime}.log
echo epoch ${e} >> result/result-${currentTime}.log
echo staleness ${staleness} >> result/result-${currentTime}.log
echo strategy ${strategy} >> result/result-${currentTime}.log

echo server info --------------------------- >> result/result-${currentTime}.log

cat logs/server.log | grep centralized | awk -F '[][]' '{print $2}' >> result/result-${currentTime}.log

echo client info ---------------------------- >> result/result-${currentTime}.log
echo >> result/result-${currentTime}.log
for(( i=0;i<${min_available_clients};i++)) do
# 	echo client${i} result >> result/result-${currentTime}.log
	ssh ${client_list[i]} "cat /opt/flower/src/py/flwr_experiment/gcn_cora/log/client0_0.log " >> result/result-${currentTime}.log
	echo , >> result/result-${currentTime}.log
done
echo >> result/result-${currentTime}.log
