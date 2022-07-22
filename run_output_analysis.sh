#to run from terminal you do 
#	chmod +x ./run_output_analysis.sh
#	./run_output_analysis.sh

for ((k=0; k<=20; k++))
do
    python run_output_analysis.py --dataset_name 'LGCP_Hawkes' --simulation_number "$k" --model_name 'LGCP' #> "output/D1/D1M1S$i.txt"
done