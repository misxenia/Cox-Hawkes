

for ((i=0; i<=99; i++))
do
    python run_inference.py --dataset_name 'LGCP_Hawkes' --simulation_number "$i" --model_name 'LGCP' --num_samples 1000 --num_warmup 500  #> "output/D1/D1M1S$i.txt"
done





#to run from terminal you do 
#	chmod +x ./run_experiment.sh
#	./run_experiment.sh