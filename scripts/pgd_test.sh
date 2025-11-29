#!/bin/bash
name="test_train_datalab"
outdir="outputs"
n_gpu=1
export DATA="/projects/m25146/data/"

    echo "Launching test for $name"
    
    sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:${n_gpu}
#SBATCH --time=00:40:00
#SBATCH --mem=256G
#SBATCH --account=m25146        # Your project account 
#SBATCH --job-name=adv_test      # Job name
#SBATCH --output=${outdir}/%x_%j.out  # Standard output and error log
#SBATCH --error=${outdir}/%x_%j.err  # Error log
source advvenv/bin/activate
# You should launch this script from the main folder
cd ./pgd
# Run your training script
python -u pgd_attack.py $@
EOT
