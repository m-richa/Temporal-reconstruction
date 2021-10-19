#!/bin/bash
#SBATCH -A richa.mishra
#SBATCH --gres=gpu:4
#SBATCH --ntasks 40
#SBATCH --mem-per-cpu=1024      
#SBATCH --mail-type=ALL
#SBATCH --mail-user=richa.mishra@research.iiit.ac.in
#SBATCH --time=2-00:00:00
#SBATCH --output=op_filetrain_rgb.txt

module load cuda/9.0
module load cudnn/7-cuda-9.0

cd
source miniconda3/etc/profile.d/conda.sh
conda activate pytorch3d

cd /ssd_scratch/cvit
echo "Date: " `date`
echo "IDs: " $CUDA_VISIBLE_DEVICES
mkdir -p /tmp/torch_richa/
export TORCH_EXTENSIONS_DIR="/tmp/torch_richa/"

#rm -rf richa
mkdir -p richa
cd richa

rsync -aP richa.mishra@ada:/share3/richa.mishra/dataset.tar.gz /ssd_scratch/cvit/richa/

tar -xzf dataset.tar.gz
echo 'Code running'

cd ~/4DReconstruction/code
python train.py --config config_ada.json
echo 'Finished'
rm -rf /tmp/torch_richa/
