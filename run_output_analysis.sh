

for ((i=0; i<=99; i++))
do
    python run_output_analysis.py --dataset_name 'LGCP-Hawkes' --simulation_number "$i" --model_name 'LGCP-Hawkes' #> "output/D1/D1M1S$i.txt"
done